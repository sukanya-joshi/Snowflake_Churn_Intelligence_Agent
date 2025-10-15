import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st

from models import (
    ChartEventData,
    DataAgentRunRequest,
    ErrorEventData,
    Message,
    MessageContentItem,
    StatusEventData,
    TableEventData,
    TextContentItem,
    TextDeltaEventData,
    ThinkingDeltaEventData,
    ThinkingEventData,
    ToolResultEventData,
    ToolUseEventData,
)

PAT = os.getenv("CORTEX_AGENT_DEMO_PAT")
HOST = os.getenv("CORTEX_AGENT_DEMO_HOST")
DATABASE = os.getenv("CORTEX_AGENT_DEMO_DATABASE", "SNOWFLAKE_INTELLIGENCE")
SCHEMA = os.getenv("CORTEX_AGENT_DEMO_SCHEMA", "AGENTS")
AGENT = os.getenv("CORTEX_AGENT_DEMO_AGENT", "SALES_INTELLIGENCE_AGENT")


def agent_run() -> requests.Response:
    """Calls the REST API and returns a streaming client."""
    request_body = DataAgentRunRequest(
        model="claude-4-sonnet",
        messages=st.session_state.messages,
    )
    resp = requests.post(
        url=f"https://{HOST}/api/v2/databases/{DATABASE}/schemas/{SCHEMA}/agents/{AGENT}:run",
        data=request_body.to_json(),
        headers={
            "Authorization": f'Bearer {PAT}"',
            "Content-Type": "application/json",
        },
        stream=True,
        verify=False,
    )
    if resp.status_code < 400:
        return resp  # type: ignore
    else:
        raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")


def stream_events(response: requests.Response):
    content = st.container()
    # Content index to container section mapping
    content_map = defaultdict(content.empty)
    # Content index to text buffer
    buffers = defaultdict(str)
    spinner = st.spinner("Waiting for response...")
    spinner.__enter__()

    events = sseclient.SSEClient(response).events()
    for event in events:
        match event.event:
            case "response.status":
                spinner.__exit__(None, None, None)
                data = StatusEventData.from_json(event.data)
                spinner = st.spinner(data.message)
                spinner.__enter__()
            case "response.text.delta":
                data = TextDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].write(buffers[data.content_index])
            case "response.thinking.delta":
                data = ThinkingDeltaEventData.from_json(event.data)
                buffers[data.content_index] += data.text
                content_map[data.content_index].expander(
                    "Thinking", expanded=True
                ).write(buffers[data.content_index])
            case "response.thinking":
                # Thinking done, close the expander
                data = ThinkingEventData.from_json(event.data)
                content_map[data.content_index].expander("Thinking").write(data.text)
            case "response.tool_use":
                data = ToolUseEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool use").json(data)
            case "response.tool_result":
                data = ToolResultEventData.from_json(event.data)
                content_map[data.content_index].expander("Tool result").json(data)
            case "response.chart":
                data = ChartEventData.from_json(event.data)
                spec = json.loads(data.chart_spec)
                content_map[data.content_index].vega_lite_chart(
                    spec,
                    use_container_width=True,
                )
            case "response.table":
                data = TableEventData.from_json(event.data)
                data_array = np.array(data.result_set.data)
                column_names = [
                    col.name for col in data.result_set.result_set_meta_data.row_type
                ]
                content_map[data.content_index].dataframe(
                    pd.DataFrame(data_array, columns=column_names)
                )
            case "error":
                data = ErrorEventData.from_json(event.data)
                st.error(f"Error: {data.message} (code: {data.code})")
                # Remove last user message, so we can retry from last successful response.
                st.session_state.messages.pop()
                return
            case "response":
                data = Message.from_json(event.data)
                st.session_state.messages.append(data)
    spinner.__exit__(None, None, None)


def process_new_message(prompt: str) -> None:
    message = Message(
        role="user",
        content=[MessageContentItem(TextContentItem(type="text", text=prompt))],
    )
    render_message(message)
    st.session_state.messages.append(message)

    with st.chat_message("assistant"):
        with st.spinner("Sending request..."):
            response = agent_run()
        st.markdown(
            f"```request_id: {response.headers.get('X-Snowflake-Request-Id')}```"
        )
        stream_events(response)


def render_message(msg: Message):
    with st.chat_message(msg.role):
        for content_item in msg.content:
            match content_item.actual_instance.type:
                case "text":
                    st.markdown(content_item.actual_instance.text)
                case "chart":
                    spec = json.loads(content_item.actual_instance.chart.chart_spec)
                    st.vega_lite_chart(spec, use_container_width=True)
                case "table":
                    data_array = np.array(
                        content_item.actual_instance.table.result_set.data
                    )
                    column_names = [
                        col.name
                        for col in content_item.actual_instance.table.result_set.result_set_meta_data.row_type
                    ]
                    st.dataframe(pd.DataFrame(data_array, columns=column_names))
                case _:
                    st.expander(content_item.actual_instance.type).json(
                        content_item.actual_instance.to_json()
                    )

from datetime import datetime, timedelta
import plotly.express as px
st.set_page_config(page_title="Cortex + Churn Dashboard", layout="wide")

# -------------------------
# Sidebar Menu
# -------------------------
st.sidebar.title("Main Menu")
page = st.sidebar.radio("Navigate", ["📈 Customer Churn Insights","📊 Marketing Dashboard","🤖 Cortex Agent"])

if page == "🤖 Cortex Agent":
    st.title("Cortex Agent")
    st.markdown("""
    Use this chat interface to ask insightful questions such as
    
    - Which customers have a high likelihood of churning and what can be done to retain them?
    - Why did Customer 4042 churn?
    - What were the main reasons customers churned?
    """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    if user_input := st.chat_input("What is your question?"):
        process_new_message(user_input)

# -------------------------
# Page 2: Churn Dashboard
# -------------------------
elif page == "📊 Marketing Dashboard":
    st.title("📊 Marketing Dashboard")
    st.markdown("""
    Use this dashboard to identify high-risk customers and take action via targeted outreach.
    Displays top churn risk segment and allows launching email campaigns.
    """)

    # -------------------------
    # Load actual churn inference data
    # -------------------------
    @st.cache_data
    def load_churn_data():
        df = pd.read_csv("/Users/sujoshi/Desktop/sfguide-getting-started-with-cortex-agents/CHURN_DATA_EXPLANATIONS.csv")
        return df

    df = load_churn_data()

    # -------------------------
    # Sidebar filters
    # -------------------------
    st.sidebar.header("🔎 Filters")
    segs = df["SUBSCRIPTION_TYPE"].unique()
    segment_filter = st.sidebar.multiselect("Subscription Type", options=segs, default=list(segs))
    threshold = st.sidebar.slider("Churn Probability ≥", 0.0, 1.0, 0.7)
    days = st.sidebar.slider("Hide recent interactions (days)", 0, 30, 15)

    filtered = df[
        (df["SUBSCRIPTION_TYPE"].isin(segment_filter)) &
        (df["CHURN_PREDICTION_PROB"] >= threshold) &
        (df["LAST_INTERACTION"] > days)
    ].sort_values(by="CHURN_PREDICTION_PROB", ascending=False)

    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("### 🔑 Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", len(df))
    c2.metric("At-Risk Customers", len(filtered))
    avg_risk = f"{filtered['CHURN_PREDICTION_PROB'].mean():.2%}" if not filtered.empty else "N/A"
    c3.metric("Avg. Pred. Churn (filtered)", avg_risk)

    st.divider()

    # -------------------------
    # At-risk Table
    # -------------------------
    st.markdown("### 🚨 High-Risk Customers")
    st.dataframe(
        filtered[[
            "CUSTOMERID", "GENDER", "CONTRACT_LENGTH", "SUBSCRIPTION_TYPE",
            "CHURN_PREDICTION_PROB", "LAST_INTERACTION", "PAYMENT_DELAY", "SUPPORT_CALLS", "TENURE", "TOTAL_SPEND"
        ]],
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # Campaign Section
    # -------------------------
    st.markdown("---")
    st.subheader("📬 Launch Targeted Campaign")

    if filtered.empty:
        st.info("No eligible customers to contact.")
    else:
        subject = st.text_input("Email Subject", "We’d love to see you stay with us")
        body = st.text_area("Email Body", "Hello, we noticed your usage has dipped—here’s an offer to stay...")
        if st.checkbox("✅ Confirm send"):
            if st.button("🚀 Send Campaign"):
                st.success(f"Sent emails to {len(filtered)} customers.")
                with st.expander("📤 Sent Emails Log"):
                    for _, row in filtered.iterrows():
                        st.markdown(f"- **{row['CUSTOMERID']}** — {row['SUBSCRIPTION_TYPE']} — Prob: {row['CHURN_PREDICTION_PROB']:.2%}")

    # -------------------------
    # Explainability / Signals
    # -------------------------
    st.markdown("### 🔍 Key Signals Driving Risk")

    st.write("""
    Based on model insights and feature attributions:
             
    • **High Support Call Volume** - Customers with 5+ support calls show the strongest churn signal, indicating unresolved issues or product dissatisfaction that requires immediate intervention

    • **Payment Delays of 15+ Days** - Consistent late payments (averaging 15-17 days across high-risk segments) signal financial stress or billing friction that can lead to cancellation

    • **Age Demographics 60+** - Older customers show significantly higher churn risk, likely due to technology adoption challenges or changing needs requiring specialized support

    • **Monthly Contract Commitment** - Month-to-month customers (especially Basic Monthly with 118 high-risk customers) demonstrate lower commitment and higher churn probability than annual subscribers

    • **Low Engagement with High Spend** - Customers spending $500+ but with low usage frequency (15-16 sessions) indicate poor value realization despite significant investment, creating churn vulnerability
        """)

    st.markdown("---")

elif page == "📈 Customer Churn Insights":
    st.title("📈 Customer Churn Insights")
    st.markdown(
        """
        Understand key churn drivers, customer behavior patterns, and opportunities for proactive retention.

        This interactive report is designed for **Customer Success**, **Product**, and **Revenue Leaders** 
        to make data-informed decisions.
        """
    )

    @st.cache_data
    def load_churn_data():
        df = pd.read_csv("/Users/sujoshi/Desktop/sfguide-getting-started-with-cortex-agents/CHURN_DATA_EXPLANATIONS.csv")
        return df

    df = load_churn_data()

    # -------------------------
    # KPI Cards
    # -------------------------
    st.markdown("## 🔑 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", len(df))
    col2.metric("⚠️ Churn Rate", f"{df['CHURN'].mean() * 100:.1f}%")
    col3.metric("💸 Avg. Total Spend", f"${df['TOTAL_SPEND'].mean():.2f}")

    st.divider()

    # -------------------------
    # Visuals
    # -------------------------
    st.markdown("## 📊 Visual Insights")

    col1, col2, col3 = st.columns(3)
    with col3:
        st.markdown("#### 🔥 Churn Probability Distribution")
        fig1 = px.histogram(
            df, x="CHURN_PREDICTION_PROB", nbins=20, 
            title="Distribution of Predicted Churn Scores", 
            color_discrete_sequence=["indianred"]
        )
        fig1.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 🧩 Subscription Type Breakdown")
        fig2 = px.pie(df, names="SUBSCRIPTION_TYPE", title="Customer Segments", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)


    with col1:
        st.markdown("#### ⏳ Tenure by Contract & Risk")
        fig4 = px.box(
            df, x="CONTRACT_LENGTH", y="TENURE", color="CHURN",
            points="all"
        )
        fig4.update_layout(xaxis_title="Contract Length", yaxis_title="Tenure (Months)")
        st.plotly_chart(fig4, use_container_width=True)


    st.divider()

    # -------------------------
    # Executive Summary
    # -------------------------
    st.markdown("## 🧠 Executive Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🚨 Critical Business Alert")
        st.markdown("- **Churn Rate:** 96% — Urgent crisis")
        st.markdown("- **Active Customers:** 40 / 1,000")
        st.markdown("- **Revenue at Risk:** $17,102 (21 at-risk)")

    with col2:
        st.markdown("### 🔍 Key Insights")
        st.markdown("- **Contracts:** 100% churn on monthly")
        st.markdown("- **Model Accuracy:** 97.8%")
        st.markdown("- **Tenure:** Avg. 30 months")
        st.markdown("- **Gender Split:** Balanced")

    with col3:
        st.markdown("### 🔥 Top Drivers")
        st.markdown("**Churn Drivers:**")
        st.markdown("- High support calls")
        st.markdown("- Late payments")
        st.markdown("- Short contracts")
  
    st.markdown("### 🔑 Recommendations")
    st.markdown("""
        To reduce churn, eliminate monthly subscription plans, which show unsustainable retention rates.  
        Deploy proactive support outreach to address issues before escalation, and prioritize retention efforts on the 21 high-risk customers identified.
    """)

    st.markdown("---")