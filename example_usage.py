from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_analyst_agent import DataAnalystAgent


if __name__ == "__main__":
    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = DataAnalystAgent(model=llm)

    # Update the CSV path for your local dataset.
    csv_path = "./data/cafe_sales.csv"

    agent.invoke_workflow(
        filepath=csv_path,
        user_instructions="Apply standard cleaning, then keep types consistent.",
    )

    cleaned_df = agent.get_data_cleaned()
    summary = agent.get_eda_summary()
    recommendations = agent.get_eda_recommendations()

    print("=== Cleaned Data Preview ===")
    if cleaned_df is not None:
        print(cleaned_df.head())
    else:
        print("No cleaned data produced.")

    print("\n=== EDA Summary ===")
    print(summary or "No summary produced.")

    print("\n=== EDA Recommendations ===")
    if recommendations:
        for i, item in enumerate(recommendations, start=1):
            print(f"{i}. {item}")
    else:
        print("No recommendations produced.")
