'''
    生成基础的nl2sql训练数据
'''
from openai import OpenAI
import json
import random

class QueryGenerator:
    def __init__(self, schema_path: str):
        with open(schema_path) as f:
            self.schema = json.load(f)
            
        # 直接使用OpenAI客户端
        self.client = OpenAI(
            api_key="sk-61e1ad520a824247b30191cd7d3bfc09",
            base_url="https://api.deepseek.com/v1"  # 注意添加/v1路径
        )

    def _build_prompt(self, db_schema) -> str:
        """构建包含多表Schema的Prompt"""
        
        # 随机选择1-3个表并获取完整schema
        # selected_tables = random.sample(list(db_schema.keys()), k=random.randint(1,3))
        # 发现模型缺乏JOIN的生成能力，因此使用2-3个表增加联表查询的训练数据
        selected_tables = random.sample(list(db_schema.keys()), k=random.randint(2,4))
        table_schemas = [db_schema[table] for table in selected_tables]
        
        # 构建多表结构描述
        schema_descriptions = []
        for i, table in enumerate(selected_tables):
            # 提取当前表信息
            cols = db_schema[table]['columns']
            
            # 格式化字段描述
            columns = "\n".join(
                [f"   - {col_name} ({info['type']})" for col_name, info in cols.items()]
            )
            
            # 构建表结构段落
            schema_desc = f"""
                            表{i+1}：{table}
                            字段列表：
                            {columns}
                        """.strip()
            schema_descriptions.append(schema_desc)
        
        return f"""
                请根据以下表结构生成SQL查询：
                一共{len(selected_tables)}张表，表名分别为：
                    {chr(10).join(selected_tables)}
                表结构（如果是多表，注意根据表之间共有的字段确定表之间的关系）：
                    {chr(10).join(schema_descriptions)}

                关于字段格式：
                (1)如果表中包含字段'交易日期'或'交易日'或'持仓日期'或'成立日期'或'到期日期'，该字段的格式形如'20231001'，请注意日期格式
                (2)如果表中包含字段'公告日期'或'截止日期'，该字段的格式形如'2023-10-01 00:00:00'，请注意日期格式
                (3)如果表中包含字段'机构投资者持有的基金份额占总份额比例'或'个人投资者持有的基金份额占总份额比例'，该字段的格式形如'99.5'，即真实值为99.5%，请注意小数格式
                (4)如果表中包含字段'定期报告所属年度'，该字段的格式形如'2023'，请注意年份格式
                (5)如果表中包含字段'持债市值占基金资产净值比'或'市值占基金资产净值比'，该字段的格式形如'0.0253'，即真实值为2.53%，请注意小数格式
                (6)如果表中包含字段'第N大重仓股'，该字段的格式形如'1'，即一个整数
                (7)如果表中包含字段'管理费率'或'托管费率'，该字段的格式形如'1.2%'

                生成要求：
                1. 如果仅有一张表，请不要强行生成联表查询语句；
                   如果有多张表，请根据表之间的关系生成联表查询语句
                    （如果所有表中的部分几张表之间有共同字段，则部分表之间可生成联表查询语句；
                      如果所有表之间均无共同字段，则不要强行生成联表查询）
                2. 请一定概率使用聚合函数、窗口函数等复杂查询语句
                3. 避免产生需要遍历全表的查询语句
                4. 请尽量产生查询效率高的SQL语句。例如：
                    - 减少子查询：用 JOIN 替换嵌套子查询
                    - 避免函数在 WHERE 条件：函数操作会阻止索引使用
                5. question的语法尽量多样化，且符合人类随意的提问口吻（即生成一定的对抗样本，挑战模型的训练）
                6. 格式规范：
                - 使用反引号包裹字段（示例：`基金类型`）
                - 表别名使用t1,t2,t3格式（示例：FROM 基金基本信息 t1）
                7. 每个查询对请包含如下字段：
                - question：自然语言问题
                - sql：对应的标准SQL语句

                生成5个question-sql查询对

                返回JSON格式：
                {{
                "queries":[
                    {{
                    "question": "问题文本",
                    "sql": "标准SQL语句"
                    }}
                ]
                }}
        """

    def generate_queries(self, db_schema) -> list:
        """生成问答对"""
        prompt = self._build_prompt(db_schema)
        
        try:
            # 调用OpenAI原生接口
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # DeepSeek指定模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=1000,
                response_format={"type": "json_object"}  # 强制JSON格式
            )
            
            # 直接解析响应内容
            content = response.choices[0].message.content
            print('content:', content)
            return json.loads(content)["queries"]
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {str(e)}")
            return []
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            return []