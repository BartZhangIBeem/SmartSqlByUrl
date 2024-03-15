from langchain_community.utilities import SQLDatabase
dburl="postgresql://118.25.188.147:5432/aidb?user=pgsql&password=pgsql"

db = SQLDatabase.from_uri(dburl)
# print(db.dialect)
# run = db.run("SELECT * FROM aiintro.EMPLOYEE LIMIT 10;")
# print(run)

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
from langchain_community.llms import Ollama
llm = Ollama(model="llama2:13b",temperature=0)
# llm = Ollama(model="gemma:7b",temperature=0)
# llm = Ollama(model="qwen:14b",temperature=0)
# llm = Ollama(model="sqlcoder",temperature=0)
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)


answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)
# chain.get_prompts()[0].pretty_print()
# invoke = chain.invoke({"question": "the schema is aiintro, the table are employee and leavebalance,"
#                                    "employee talbe and leavebalance table is connect by their emmployeeid column,"
#                                    "pls give me all employee's leave balance info"})
invoke = chain.invoke({"question": "在aiintro的schema里面,有两个表 employee and leavebalance,"
                                   "这里两个表中 emmployeeid 列是互相关联的,"
                                   "请输出employee表中,所有employee的详细leavebalance信息"})
print(invoke)