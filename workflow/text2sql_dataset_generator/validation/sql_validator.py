
from sqlglot import parse, exp
from sqlglot.errors import ParseError

class SQLValidator:
    def __init__(self, schema: dict):
        self.schema = schema
    
    def validate_syntax(self, sql: str) -> bool:
        """语法校验"""
        try:
            parse(sql)
            return True
        except ParseError:
            return False
    
    def validate_semantic(self, sql: str) -> bool:
        """语义校验"""
        try:
            for table in parse(sql).find_all(exp.Table):
                if table.name not in self.schema:
                    print(f"表 {table.name} 不在数据库模式中")
                    return False
            return True
        except:
            return False