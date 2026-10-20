# SPDX-License-Identifier: Apache-2.0
"""CPU-only tuning-table reflection tests; no GPU backend initialization."""

import ast
from pathlib import Path

import sqlalchemy as sa
import sqlalchemy.ext.automap


def load_reflector():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/flag_gems/utils/models/sql.py"
    )
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SQLPersistantModel"
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_sql_model_by_db"
    )
    fn.decorator_list = []
    namespace = {"sqlalchemy": sa}
    exec(
        "from __future__ import annotations\n" + ast.unparse(fn),
        namespace,
    )
    return namespace[fn.name]


def collect_statements(engine):
    statements = []

    def record(conn, cursor, statement, parameters, context, many):
        statements.append(statement)

    sa.event.listen(engine, "before_cursor_execute", record)
    return statements


def test_existing_composite_key_and_configuration():
    engine = sa.create_engine("sqlite:///:memory:")
    meta = sa.MetaData()
    target = sa.Table(
        "kernel-tune-with-hyphens",
        meta,
        sa.Column("key_0", sa.BigInteger, primary_key=True),
        sa.Column("key_1", sa.String, primary_key=True),
        sa.Column("BLOCK_M", sa.Integer),
        sa.Column("flag", sa.Boolean),
        sa.Column("optional", sa.Float, nullable=True),
    )
    for index in range(12):
        sa.Table(
            f"unrelated_{index}",
            meta,
            sa.Column("id", sa.Integer, primary_key=True),
        )
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            target.insert(),
            [
                dict(
                    key_0=2**40,
                    key_1="bf16",
                    BLOCK_M=128,
                    flag=True,
                    optional=None,
                )
            ],
        )

    statements = collect_statements(engine)
    model = load_reflector()(target.name, engine)
    assert model is not None
    assert list(model.__table__.primary_key.columns.keys()) == ["key_0", "key_1"]
    with engine.connect() as conn:
        got = conn.execute(sa.select(model.__table__)).one()
    assert tuple(got) == (2**40, "bf16", 128, True, None)
    assert not any(
        "table_info" in statement and "unrelated_" in statement
        for statement in statements
    )


def test_missing_table_does_not_reflect_unrelated_tables():
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("create table existing (id integer primary key)")
    statements = collect_statements(engine)
    assert load_reflector()("missing", engine) is None
    assert not any("PRAGMA" in statement for statement in statements)


def test_empty_database_and_unmapped_table():
    engine = sa.create_engine("sqlite:///:memory:")
    assert load_reflector()("absent", engine) is None
    with engine.begin() as conn:
        conn.exec_driver_sql("create table no_primary_key (value integer)")
    assert load_reflector()("no_primary_key", engine) is None


if __name__ == "__main__":
    test_existing_composite_key_and_configuration()
    test_missing_table_does_not_reflect_unrelated_tables()
    test_empty_database_and_unmapped_table()
    print("3 CPU reflection regression tests passed")
