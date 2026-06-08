from setuptools import find_packages, setup


setup(
    name="simple-langgraph-agent",
    version="0.1.0",
    description="A tiny LangGraph agent with optional Serper web search.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastapi>=0.110.0",
        "langchain-openai>=0.1.0",
        "langgraph>=0.2.0",
        "python-dotenv>=1.0.0",
        "uvicorn[standard]>=0.27.0",
    ],
    entry_points={
        "console_scripts": [
            "simple-agent=simple_langgraph_agent.agent:main",
        ],
    },
    python_requires=">=3.9",
)
