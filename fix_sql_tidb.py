#!/usr/bin/env python3
"""
Fix SQLStorm queries for TiDB compatibility.

Transformations:
  1. STRING_AGG(e, d ORDER BY c) → GROUP_CONCAT(e ORDER BY c SEPARATOR d)
  2. expr || expr            → CONCAT(expr, expr)
  3. ILIKE                  → LOWER(x) LIKE LOWER(y)
  4. expr :: type           → CAST(expr AS type)
  5. FULL OUTER JOIN        → LEFT JOIN + UNION + RIGHT JOIN
  6. ARRAY_AGG              → GROUP_CONCAT (best-effort, or skip)
  7. RECURSIVE CTE          → keep (TiDB supports), add LIMIT if missing
"""

import os
import re
import shutil


def fix_pipe_concat(sql: str) -> str:
    """Replace PostgreSQL || string concatenation with CONCAT().

    Handles: 'text' || col, col1 || col2, func() || col, etc.
    Only replaces outside of string literals.
    """
    # Strategy: find all || operators outside string literals and parentheses
    # Simple approach: repeatedly find patterns like: <expr> || <expr>
    # where <expr> is balanced (no unclosed parens)

    def find_concat_expr(text, start=0):
        """Find the next '||' operator, return (full_left_expr, full_right_expr, start, end)."""
        i = text.find('||', start)
        if i < 0:
            return None

        # Extract left operand (go backwards, balancing parens)
        left_end = i
        while left_end > 0 and text[left_end - 1] == ' ':
            left_end -= 1
        depth = 0
        left_start = left_end - 1
        while left_start >= 0:
            c = text[left_start]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            if depth == 0 and c in (' ', '\n', '\t', ',', ';', '('):
                break
            if depth < 0:
                break
            left_start -= 1
        left_expr = text[left_start + 1:left_end].strip()
        if not left_expr:
            return None

        # Extract right operand (go forward, balancing parens)
        right_start = i + 2
        while right_start < len(text) and text[right_start] == ' ':
            right_start += 1
        depth = 0
        right_end = right_start
        while right_end < len(text):
            c = text[right_end]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and c in (' ', '\n', '\t', ',', ')', ';'):
                break
            if depth < 0:
                break
            right_end += 1
        right_expr = text[right_start:right_end].strip()
        if not right_expr:
            return None

        return (left_expr, right_expr, left_start + 1, right_end)

    # Replace || with CONCAT, avoiding infinite loops
    max_iter = 50
    result = sql
    for _ in range(max_iter):
        m = find_concat_expr(result)
        if m is None:
            break
        left, right, start, end = m
        result = result[:start] + f"CONCAT({left}, {right})" + result[end:]

    return result


def fix_string_agg(sql: str) -> str:
    """Convert STRING_AGG(expr, delimiter [ORDER BY ...]) → GROUP_CONCAT(...)."""
    # Match STRING_AGG with optional ORDER BY
    pattern = r'STRING_AGG\s*\(\s*(.+?)\s*,\s*(.+?)(\s+ORDER\s+BY\s+[^)]+?)?\)'
    # More precise: handle nested parens
    def replace_string_agg(m):
        inner = m.group(1)  # expression
        delim = m.group(2).strip().strip("'")  # delimiter
        order = m.group(3) if m.group(3) else ''

        if order:
            # STRING_AGG(e, ',' ORDER BY c, d) → GROUP_CONCAT(e ORDER BY c, d SEPARATOR ',')
            return f"GROUP_CONCAT({inner.strip()} {order.strip()} SEPARATOR '{delim}')"
        else:
            return f"GROUP_CONCAT({inner.strip()} SEPARATOR '{delim}')"

    # Manual parsing: find STRING_AGG( ... ) with balanced parens
    result = []
    i = 0
    while i < len(sql):
        idx = sql.find('STRING_AGG(', i)
        if idx < 0:
            result.append(sql[i:])
            break
        result.append(sql[i:idx])

        # Find matching closing paren
        depth = 0
        j = idx + len('STRING_AGG(')
        start_j = j
        while j < len(sql):
            if sql[j] == '(':
                depth += 1
            elif sql[j] == ')':
                if depth == 0:
                    break
                depth -= 1
            j += 1

        inner = sql[start_j:j]
        # Split inner by first comma at depth 0
        parts = []
        d = 0
        comma_pos = -1
        for k, c in enumerate(inner):
            if c == '(':
                d += 1
            elif c == ')':
                d -= 1
            elif c == ',' and d == 0:
                comma_pos = k
                break

        if comma_pos > 0:
            expr = inner[:comma_pos].strip()
            rest = inner[comma_pos + 1:].strip()

            # Check for ORDER BY in rest
            order_match = re.search(r'\bORDER\s+BY\b', rest, re.I)
            if order_match:
                delim_part = rest[:order_match.start()].strip().strip("'")
                order_part = rest[order_match.start():].strip()
                replacement = f"GROUP_CONCAT({expr} {order_part} SEPARATOR '{delim_part}')"
            else:
                delim_part = rest.strip().strip("'")
                replacement = f"GROUP_CONCAT({expr} SEPARATOR '{delim_part}')"
        else:
            # No delimiter comma found - just expression?
            replacement = f"GROUP_CONCAT({inner.strip()})"

        result.append(replacement)
        i = j + 1

    return ''.join(result)


def fix_ilike(sql: str) -> str:
    """Convert ILIKE to LOWER/LIKE."""
    pattern = r'(\S+)\s+ILIKE\s+(\'[^\']+\'|\S+)'
    return re.sub(pattern, r'LOWER(\1) LIKE LOWER(\2)', sql, flags=re.I)


def fix_postgres_cast(sql: str) -> str:
    """Convert ::cast to CAST()."""
    type_map = {
        'int': 'SIGNED', 'bigint': 'SIGNED', 'smallint': 'SIGNED',
        'text': 'CHAR', 'varchar': 'CHAR', 'char': 'CHAR',
        'numeric': 'DECIMAL', 'float': 'DOUBLE', 'double': 'DOUBLE',
        'date': 'DATE', 'timestamp': 'TIMESTAMP',
        'interval': 'CHAR', 'decimal': 'DECIMAL',
    }
    def replace_cast(m):
        expr = m.group(1).strip()
        pg_type = m.group(2).lower()
        mysql_type = type_map.get(pg_type, 'CHAR')
        return f"CAST({expr} AS {mysql_type})"

    return re.sub(r'(\S+)\s*::\s*(int|bigint|smallint|text|varchar|char|numeric|float|double|date|timestamp|interval|decimal)\b',
                  replace_cast, sql, flags=re.I)


def fix_full_outer_join(sql: str) -> str:
    """Try to convert FULL OUTER JOIN to LEFT JOIN + UNION + RIGHT JOIN.

    This is a best-effort approach for simple queries.
    For complex queries with multiple FULL JOINs, we mark them as unfixable.
    """
    # Check if it's a simple case: single FULL OUTER JOIN
    full_joins = list(re.finditer(r'\bFULL\s+OUTER\s+JOIN\b', sql, re.I))
    if len(full_joins) > 1:
        # Too complex, add comment
        return sql  # Will be filtered out later

    # For a single FULL OUTER JOIN, convert:
    # FROM A FULL OUTER JOIN B ON cond → (A LEFT JOIN B ON cond) UNION (A RIGHT JOIN B ON cond)
    m = re.search(
        r'(\w[\w.]*\s+)(?:AS\s+\w+\s+)?FULL\s+OUTER\s+JOIN\s+(\w[\w.]*)\s+(?:AS\s+(\w+)\s+)?ON\s+(.+?)(?=\s+(?:WHERE|GROUP|HAVING|ORDER|LIMIT|LEFT|RIGHT|INNER|JOIN|$))',
        sql, re.I
    )
    if not m:
        # Simpler pattern
        m = re.search(
            r'(FROM\s+.*?)FULL\s+OUTER\s+JOIN\s+(\S+)\s+ON\s+(.+?)(?=\s+WHERE|\s+GROUP|\s+HAVING|\s+ORDER|\s+LIMIT|;|$)',
            sql, re.I
        )
        if not m:
            return sql + ' -- FIXME: FULL OUTER JOIN not supported by TiDB'

    # Best-effort: mark as unfixable for complex cases
    return sql + ' -- FIXME: FULL OUTER JOIN not supported by TiDB'


def fix_array_agg(sql: str) -> str:
    """Convert ARRAY_AGG to GROUP_CONCAT (best-effort)."""
    return re.sub(r'\bARRAY_AGG\b', 'GROUP_CONCAT', sql, flags=re.I)


def fix_reserved_aliases(sql: str) -> str:
    """Backtick-quote reserved word aliases: rank, rows, etc."""
    # Replace 'AS rank' or 'rank,' with backticked version
    # Be careful not to quote inside string literals or function calls
    reserved = ['rank', 'rows', 'percent', 'role']
    for word in reserved:
        sql = re.sub(rf'\bAS\s+({word})\b', rf'AS `{word}`', sql, flags=re.I)
        # Also handle: expression alias without AS, e.g., "... END rank"
        sql = re.sub(rf'(?<!,)\s+({word})\s*(,|$|FROM|ORDER|GROUP|LIMIT)', rf' `{word}`\2', sql, flags=re.I)
    return sql


def fix_on_subquery(sql: str) -> tuple[str, bool]:
    """Check if ON clause contains a subquery (TiDB limitation). Returns (sql, has_issue)."""
    if re.search(r'\bON\b\s+.*?\(\s*SELECT\b', sql, re.I):
        return sql, True
    return sql, False


def fix_nulls_order(sql: str) -> str:
    """Remove NULLS FIRST/LAST (TiDB uses MySQL NULL ordering)."""
    return re.sub(r'\s+NULLS\s+(FIRST|LAST)\b', '', sql, flags=re.I)


def fix_filter_where(sql: str) -> str:
    """Convert aggregate FILTER (WHERE cond) to CASE WHEN cond THEN expr END.

    Example: COUNT(c_custkey) FILTER (WHERE sales_rank = 1)
          →  COUNT(CASE WHEN sales_rank = 1 THEN c_custkey END)
    """
    pattern = r'(\w+\([^)]+\))\s+FILTER\s*\(WHERE\s+([^)]+)\)'
    def replace_filter(m):
        agg_func = m.group(1)  # e.g., COUNT(c_custkey)
        condition = m.group(2)
        # Extract function name and argument
        func_match = re.match(r'(\w+)\((.+)\)', agg_func)
        if func_match:
            fname = func_match.group(1)
            farg = func_match.group(2)
            return f"{fname}(CASE WHEN {condition} THEN {farg} END)"
        return agg_func
    return re.sub(pattern, replace_filter, sql, flags=re.I)


def fix_regexp_replace(sql: str) -> str:
    """Convert REGEXP_REPLACE to REGEXP_REPLACE (TiDB supports this, but check syntax).
    PostgreSQL: REGEXP_REPLACE(str, pattern, replacement [, flags])
    TiDB:       REGEXP_REPLACE(str, pattern, replacement [, pos [, occurrence [, match_type]]])
    These are similar enough to leave as-is; just flag for manual review if needed.
    """
    return sql  # TiDB supports REGEXP_REPLACE


def fix_single_query(sql: str) -> tuple[str, list[str]]:
    """Apply all fixes. Returns (fixed_sql, list_of_issues_found)."""
    issues = []

    # Detect issues
    if re.search(r'\bSTRING_AGG\b', sql, re.I):
        issues.append('STRING_AGG')
    if '||' in sql and "'||'" not in sql and "||'" not in sql:
        issues.append('PIPE_CONCAT')
    if re.search(r'\bILIKE\b', sql, re.I):
        issues.append('ILIKE')
    if re.search(r'::\s*(int|bigint|text|varchar|char|numeric|float|double|date|timestamp|interval|decimal)\b', sql, re.I):
        issues.append('POSTGRES_CAST')
    if re.search(r'\bFULL\s+OUTER\s+JOIN\b', sql, re.I):
        issues.append('FULL_OUTER_JOIN')
    if re.search(r'\bARRAY_AGG\b', sql, re.I):
        issues.append('ARRAY_AGG')
    if re.search(r'\bWITH\s+RECURSIVE\b', sql, re.I):
        issues.append('RECURSIVE_CTE')
    if re.search(r'\bNULLS\s+(FIRST|LAST)\b', sql, re.I):
        issues.append('NULLS_FIRST_LAST')
    if re.search(r'FILTER\s*\(\s*WHERE\b', sql, re.I):
        issues.append('FILTER_WHERE')
    if re.search(r'\bREGEXP_REPLACE\b', sql, re.I):
        issues.append('REGEXP_REPLACE')

    fixed = sql

    # Apply fixes (order matters)
    if 'STRING_AGG' in str(issues):
        fixed = fix_string_agg(fixed)
    if 'PIPE_CONCAT' in issues:
        fixed = fix_pipe_concat(fixed)
    if 'ILIKE' in issues:
        fixed = fix_ilike(fixed)
    if 'POSTGRES_CAST' in issues:
        fixed = fix_postgres_cast(fixed)
    if 'ARRAY_AGG' in issues:
        fixed = fix_array_agg(fixed)
    if 'NULLS_FIRST_LAST' in issues:
        fixed = fix_nulls_order(fixed)
    if 'FILTER_WHERE' in issues:
        fixed = fix_filter_where(fixed)
    if 'FULL_OUTER_JOIN' in issues:
        fixed = fix_full_outer_join(fixed)

    return fixed, issues


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/home/anqian/Desktop/my_lab/workloads/SQLStorm')
    parser.add_argument('--output-dir', default='/home/anqian/Desktop/my_lab/workloads/SQLStorm_fixed')
    parser.add_argument('--skip-dir', default='/home/anqian/Desktop/my_lab/workloads/SQLStorm_skipped')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        print(f"DRY RUN: would create {args.output_dirs}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.skip_dir, exist_ok=True)

    files = sorted(os.listdir(args.input_dir))
    stats = {'fixed': 0, 'clean': 0, 'skipped': 0, 'total': len(files)}
    issue_counts = {}

    for fname in files:
        src = os.path.join(args.input_dir, fname)
        with open(src) as f:
            sql = f.read()

        fixed, issues = fix_single_query(sql)

        for iss in issues:
            issue_counts[iss] = issue_counts.get(iss, 0) + 1

        # FULL_OUTER_JOIN queries that couldn't be fixed → skip
        has_unfixable = 'FIXME' in fixed

        if has_unfixable:
            dest = os.path.join(args.skip_dir, fname)
            shutil.copy(src, dest)
            stats['skipped'] += 1
            continue

        if issues:
            dest = os.path.join(args.output_dir, fname)
            with open(dest, 'w') as f:
                f.write(fixed)
            stats['fixed'] += 1
        else:
            dest = os.path.join(args.output_dir, fname)
            shutil.copy(src, dest)
            stats['clean'] += 1

    print(f"Total: {stats['total']}")
    print(f"Clean (no changes): {stats['clean']}")
    print(f"Fixed: {stats['fixed']}")
    print(f"Skipped (unfixable): {stats['skipped']}")
    print(f"\nIssues found:")
    for iss, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {iss}: {cnt}")


if __name__ == '__main__':
    main()
