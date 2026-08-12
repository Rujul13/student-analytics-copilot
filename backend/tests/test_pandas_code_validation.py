import pytest

from app.pandas_code_validation import CodeValidationError, validate_code


def test_accepts_a_valid_scalar_program():
    code = "result = float(enrollments['course_code'].eq('BBB').sum())"
    validate_code(code)  # must not raise


def test_accepts_a_valid_groupby_table_program():
    code = (
        "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
        "grouped = merged.groupby('course_code').agg(\n"
        "    enrollments_count=('enrollment_id', 'count'),\n"
        "    withdrawals=('final_result', lambda values: (values == 'Withdrawn').sum()),\n"
        ")\n"
        "grouped['withdrawal_rate'] = (grouped['withdrawals'] / grouped['enrollments_count'] * 100).round(1)\n"
        "result = grouped.sort_values('withdrawal_rate', ascending=False).reset_index().head(100)\n"
    )
    validate_code(code)  # must not raise


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1",
        "from os import path\nresult = 1",
        "result = open('secret.csv')",
        "result = eval('1+1')",
        "result = exec('x=1')",
        "result = __import__('os')",
        "result = compile('1', 'f', 'eval')",
        "x = input()\nresult = x",
        "result = globals()",
        "result = locals()",
        "result = vars()",
        "result = getattr(enrollments, 'to_csv')",
        "import os as o\nresult = o.getcwd()",
        "result = enrollments.__class__",
        "result = os.system('dir')",
        "result = subprocess.run(['dir'])",
        "def helper():\n    return 1\nresult = helper()",
        "class Helper:\n    pass\nresult = 1",
        "result = 1\nwhile True:\n    pass",
        "with open('f') as fh:\n    result = 1",
        "try:\n    result = 1\nexcept Exception:\n    result = 2",
        "result = 1\nraise ValueError('x')",
        "def gen():\n    yield 1\nresult = list(gen())",
        "async def f():\n    return 1\nresult = 1",
        "enrollments['x'] = 1\nresult = enrollments",
        "enrollments.loc[0, 'x'] = 1\nresult = enrollments",
        "del enrollments['course_code']\nresult = 1",
        "students = students.head(1)\nresult = students",
        "enrollments.to_csv('out.csv')\nresult = 1",
        "result = pd.read_csv('x.csv')",
        "result = enrollments.to_pickle('x.pkl')",
        "enrollments.drop(columns=['course_code'], inplace=True)\nresult = 1",
    ],
)
def test_rejects_unsafe_or_disallowed_code(code):
    with pytest.raises(CodeValidationError):
        validate_code(code)


def test_requires_a_result_assignment():
    with pytest.raises(CodeValidationError, match="result"):
        validate_code("value = enrollments['course_code'].nunique()")


def test_rejects_code_over_the_length_limit():
    with pytest.raises(CodeValidationError, match="length"):
        validate_code("result = 1  # " + "x" * 5000, max_length=4000)


def test_rejects_unparseable_code():
    with pytest.raises(CodeValidationError):
        validate_code("result = (")
