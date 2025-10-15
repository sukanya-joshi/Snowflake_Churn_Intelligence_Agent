# Churn Intelligence Agent using Snowflake ML and Snowflake Intelligence
This is a project which combines Snowflake ML and AI to build a churn intelligence agent.
Step-By-Step Guide
For the easiest implementation to get the streamlit application running (without the ML implementation):

1. Upload CHURN_DATA_EXPLANATIONS.csv into a Snowflake table.

2. Create a Semantic View called "CHURN_EXPLANATION_TOOL" on top of your Snowflake table by going to AI/ML -> Cortex Analyst

3. Create a PAT token to call the Snowflake REST APIs
In Snowsight:

Click on your profile (bottom left corner) » Settings » Authentication
Under Programmatic access tokens, click Generate new token
Copy and save the token for later (you will not be able to see it again)

4. Create the Agent
   
Create the Agent in Snowsight navigating to AI & ML » Agents » Create Agent (see step by step in the QuickStart Guide). Name the agent as "CHURN_INTELLIGENCE_AGENT" and use the CHURN_EXPLANATION_TOOL semantic view as your tool under the Cortex Analyst section.

After creating you can chat with the Agent via Snowflake Intelligence. In Snowsight, click on AI & ML » Snowflake Intelligence, select the CHURN_INTELLIGENCE_AGENT in the chat bar, and ask any questions you'd like!

5. Run the streamlit
Now that you have created your agent, run the sample streamlit to interact with it. First, clone this repository in your local VSCode. Then, make sure to populate the PAT and HOST params correctly. (Note: this guide was built using Python 3.11). Be sure to modify your data_agent_demo.py to include your own file path for the csv.

```bash
# 1. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install required packages
pip3 install -r requirements.txt

# 3. Run Streamlit app with required environment variables
CORTEX_AGENT_DEMO_PAT=<PAT> \
CORTEX_AGENT_DEMO_HOST=<ACCOUNT_URL> \
CORTEX_AGENT_DEMO_DATABASE="SNOWFLAKE_INTELLIGENCE" \
CORTEX_AGENT_DEMO_SCHEMA="AGENTS" \
CORTEX_AGENT_DEMO_AGENT="CHURN_INTELLIGENCE_AGENT" \
streamlit run data_agent_demo.py
```

The streamlit uses python code auto-generated using https://openapi-generator.tech/. It creates the pydantic classes in /models for the request and response objects based on the OpenAPI spec at cortexagent-run.yaml. You can regenerate those files by running the script openapi-generator.sh (assuming you have docker installed and running locally).

Streamlit App Screenshots:
<img width="1694" height="817" alt="Screenshot 2025-10-15 at 3 10 44 PM" src="https://github.com/user-attachments/assets/86171dc7-83dc-4f87-97af-f38ea562fdd1" />

<img width="1696" height="908" alt="Screenshot 2025-10-15 at 3 11 06 PM" src="https://github.com/user-attachments/assets/16c30b4d-bb76-4e68-919b-135ad4e9c075" />

<img width="1693" height="831" alt="Screenshot 2025-10-15 at 3 12 08 PM" src="https://github.com/user-attachments/assets/58a54266-92ea-4b32-afa2-d0915dc711c0" />
<img width="1691" height="893" alt="Screenshot 2025-10-15 at 3 12 22 PM" src="https://github.com/user-attachments/assets/0613421a-c7a7-4be5-a7ac-51038a17a69a" />

