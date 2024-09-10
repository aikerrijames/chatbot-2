# Importing packages

import streamlit as st
from openai import OpenAI
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Dict, Any
from functools import partial

# Initializing the streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Credentials are saved as streamlit secrets in TOML format so they don't go into github

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"])

client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Tools will be defined below. The agent will have access to these tools in order to answer the user's questions.

# The DataSchemaRoadmapTool contains the shape of each of the looker Studio related tables. The agent will consult this tool first, in order to decide which table to query.
# Any new tables to be queried should be added to the DataSchemaRoadmapTool, with their respective schemas

class DataSchemaRoadmapTool(BaseTool):
    name: str = "Data Schema Roadmap"
    description: str = "Provides information about the data schema roadmap for Looker Studio visualizations"
    schema_map: dict = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        self.schema_map = {
            "the_pulse.reviews_forLS": {
                "description": "Contains detailed review information",
                "fields": ["id", "ghl_location_name", "ghl_location_id", "comments", "rating", "reviewer", "google_review_id", "review_added_on", "reviews_conversation", "is_posted_on_fb", "user_id", "review_id", "locationid", "review_path", "ai_generated_response", "createdAt", "updatedAt", "posted_on_fb", "posted_on_insta", "posted_on_other", "posted_on_twitter", "email", "five_star_rating", "one_to_four_star_rating", "rating_2", "rating_3"]
            },
            "the_pulse.opportunities_monthly_forLS": {
                "description": "Monthly aggregated data on sales opportunities",
                "fields": ["ghl_location_id", "ghl_location_name", "opportunity_created_date", "Month", "Leads", "Qualified", "Retained", "Lost", "Signed_Cases", "Revenue"]
            },
            "the_pulse.calls_monthly_forLS": {
                "description": "Monthly aggregated call data",
                "fields": ["ghl_location_id", "ghl_location_name", "call_date", "Month", "Total_Calls", "Answered_Calls"]
            },
            "the_pulse.calls_forLS": {
                "description": "Detailed call data",
                "fields": ["ghl_location_id", "ghl_location_name", "call_date", "Total_Calls", "Answered_Calls", "Missed_Calls"]
            },
            "the_pulse.ad_expense_data_monthly_forLS": {
                "description": "Monthly aggregated ad expense data",
                "fields": ["ghl_location_id", "ghl_location_name", "formatted_expense_monthyear", "Month", "ad_expense"]
            },
            "the_pulse.ad_expense_data_forLS": {
                "description": "Detailed ad expense data",
                "fields": ["ghl_location_id", "ghl_location_name", "id", "ad_expense", "expense_monthyear", "expense_month", "expense_year", "converted_leads", "forecasted_revenue", "created_on", "updated_on", "formatted_expense_monthyear"]
            },
            "the_pulse.opportunities_forLS": {
                "description": "Detailed information on sales opportunities",
                "fields": ["ghl_location_id", "ghl_location_name", "contact_id", "contact_email", "contact_tags", "opportunity_created_date", "opportunity_id", "Leads", "Qualified", "Retained", "Open_1", "Lost", "Signed_Cases", "opportunity_pipeline_name", "opportunity_pipeline_id", "opportunity_stage_name", "opportunity_stage_id", "opportunity_source", "Revenue", "Status", "opportunity_assigned_user", "Location", "Medium", "Type"]
            }
        }

    def _run(self, query: str) -> str:
        if query.lower() == "list tables":
            return "\n".join(self.schema_map.keys())
        elif query.lower().startswith("describe "):
            table_name = query.split(" ", 1)[1]
            if table_name in self.schema_map:
                table_info = self.schema_map[table_name]
                return f"Table: {table_name}\nDescription: {table_info['description']}\nFields: {', '.join(table_info['fields'])}"
            else:
                return f"Table {table_name} not found in the schema map."
        else:
            return "Invalid query. Use 'list tables' to see all tables or 'describe [table_name]' for details on a specific table."

    async def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async operations")

# The bigquery tool is responsible for actually sending the SQL query to BigQuery and obtaining a response.
# This tool receives the SQL built by the agent after the agent used the other tools. If filtering by any field is necessary, I'd recommend adding it to the BigQuery tool so that it is "hardcoded" into the query (eg: the agent provides a query that goes "SELECT * from calls_forLS", and the bigquery tool forcibly adds "WHERE ghl_location_id is {location_id}", with location_id as a placeholder for a client name you can pass as a variable). This would make sure the query regards the client.

def execute_bigquery(client: Any, query: str) -> str:
    try:
        query_job = client.query(query)
        results = query_job.result()
        return str([dict(row) for row in results])
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Each table gets its own tool which contains the table name, description, each table field and how it was calculated, and the whole query that creates the table in question. The agent will consult the tool in order to mimic the exact query that generated the data which the client is asking for. This was done in order to make the agent reply match the looker studio data. Rather than creating its own query, the agent will copy the query which was used to build the LS tables. This should return the same results as the dashboard as long as no special filters are applied in the dashboard itself.

class CallsForLSTool(BaseTool):
    name: str = "Calls_ForLS"
    description: str = "Provides information about call data from the calls_forLS table."

    @property
    def table_info(self):
        return {
            "name": "calls_forLS",
            "description": "Provides information about the structure and calculations in the calls_forLS table.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "call_date": "Calculated using: PARSE_DATE('%Y-%m-%d', SUBSTR(call_started_at, 1, 10)) AS call_date",
                "Total_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" THEN 1 ELSE 0 END) AS Total_Calls",
                "Answered_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" AND call_status = \"completed\" THEN 1 ELSE 0 END) AS Answered_Calls",
                "Missed_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" AND call_status = \"no-answer\" THEN 1 ELSE 0 END) AS Missed_Calls"
            },
            "source_table": "the-pulse-405018.the_pulse.calls_bq",
            "clustering": "CLUSTER BY ghl_location_id",
            "grouping": "GROUP BY ghl_location_id, ghl_location_name, call_started_at",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.calls_forLS`
            CLUSTER BY ghl_location_id AS
            SELECT
                ghl_location_id,
                ghl_location_name,
                PARSE_DATE('%Y-%m-%d', SUBSTR(call_started_at, 1, 10)) AS call_date,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" THEN 1 ELSE 0 END) AS Total_Calls,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" AND call_status = "completed" THEN 1 ELSE 0 END) AS Answered_Calls,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" AND call_status = "no-answer" THEN 1 ELSE 0 END) AS Missed_Calls
            FROM `the-pulse-405018.the_pulse.calls_bq`
            GROUP BY ghl_location_id, ghl_location_name, call_started_at
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Follows the same pattern as the other tables

class AdExpenseDataForLSTool(BaseTool):
    name: str = "Ad_Expense_Data_ForLS"
    description: str = "Provides information about the structure and calculations in the ad_expense_data_forLS table."

    @property
    def table_info(self):
        return {
            "name": "ad_expense_data_forLS",
            "description": "Provides information about ad expense data.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "formatted_expense_monthyear": "Calculated using: DATE_TRUNC(expense_date, MONTH) AS formatted_expense_monthyear",
                "ad_expense": "Directly selected from the source table using: SELECT ad_expense"
            },
            "source_table": "the-pulse-405018.the_pulse.ad_expense_data",
            "partitioning": "PARTITION BY formatted_expense_monthyear",
            "clustering": "CLUSTER BY ghl_location_id",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.ad_expense_data_forLS`
            CLUSTER BY ghl_location_id, ghl_location_name AS
            SELECT
                ghl_location_id,
                ghl_location_name,
                id,
                ad_expense,
                expense_monthyear,
                expense_month,
                expense_year,
                converted_leads,
                forecasted_revenue,
                created_on,
                updated_on,
                PARSE_DATE('%Y-%m-%d',
                    CONCAT(
                        RIGHT(expense_monthyear, 4),
                        '-',
                        LPAD(
                            FORMAT_DATE('%m', PARSE_DATE('%B', LEFT(expense_monthyear, LENGTH(expense_monthyear) - 5))),
                            2,
                            '0'
                        ),
                        '-01'
                    )
                ) AS formatted_expense_monthyear
            FROM `the-pulse-405018.the_pulse.ad_expense_data_bq`
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Follows the same pattern as the other tables

class ReviewsForLSTool(BaseTool):
    name: str = "Reviews_ForLS"
    description: str = "Provides information about the structure and calculations in the reviews_forLS table."

    @property
    def table_info(self):
        return {
            "name": "reviews_forLS",
            "description": "Provides information about the structure and calculations in the reviews_forLS table.",
            "fields": {
                "id": "Directly selected from the source table using: SELECT id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "comments": "Directly selected from the source table using: SELECT comments",
                "rating": "Directly selected from the source table using: SELECT rating",
                "reviewer": "Directly selected from the source table using: SELECT reviewer",
                "google_review_id": "Directly selected from the source table using: SELECT google_review_id",
                "review_added_on": "Directly selected from the source table using: SELECT review_added_on",
                "reviews_conversation": "Directly selected from the source table using: SELECT reviews_conversation",
                "is_posted_on_fb": "Directly selected from the source table using: SELECT is_posted_on_fb",
                "user_id": "Directly selected from the source table using: SELECT user_id",
                "review_id": "Directly selected from the source table using: SELECT review_id",
                "locationid": "Directly selected from the source table using: SELECT locationid",
                "review_path": "Directly selected from the source table using: SELECT review_path",
                "ai_generated_response": "Directly selected from the source table using: SELECT ai_generated_response",
                "createdAt": "Directly selected from the source table using: SELECT createdAt",
                "updatedAt": "Directly selected from the source table using: SELECT updatedAt",
                "posted_on_fb": "Directly selected from the source table using: SELECT posted_on_fb",
                "posted_on_insta": "Directly selected from the source table using: SELECT posted_on_insta",
                "posted_on_other": "Directly selected from the source table using: SELECT posted_on_other",
                "posted_on_twitter": "Directly selected from the source table using: SELECT posted_on_twitter",
                "email": "Directly selected from the source table using: SELECT email",
                "five_star_rating": "Calculated using: CASE WHEN rating = \"FIVE\" THEN 1 ELSE 0 END AS five_star_rating",
                "one_to_four_star_rating": "Calculated using: CASE WHEN (rating != \"FIVE\" AND rating != \"ZERO\") THEN 1 ELSE 0 END AS one_to_four_star_rating",
                "rating_2": "Calculated using: CASE WHEN rating = \"FIVE\" THEN 5 WHEN rating = \"FOUR\" THEN 4 WHEN rating = \"THREE\" THEN 3 WHEN rating = \"TWO\" THEN 2 WHEN rating = \"ONE\" THEN 1 WHEN rating = \"ZERO\" THEN 0 ELSE NULL END AS rating_2",
                "rating_3": "Calculated using: CASE WHEN rating = \"FIVE\" THEN \"5 star reviews\" ELSE \"others\" END AS rating_3"
            },
            "source_table": "the-pulse-405018.the_pulse.reviews_bq",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.reviews_forLS` AS
            SELECT
                id,
                ghl_location_name,
                ghl_location_id,
                comments,
                rating,
                reviewer,
                google_review_id,
                review_added_on,
                reviews_conversation,
                is_posted_on_fb,
                user_id,
                review_id,
                locationid,
                review_path,
                ai_generated_response,
                createdAt,
                updatedAt,
                posted_on_fb,
                posted_on_insta,
                posted_on_other,
                posted_on_twitter,
                email,
                CASE WHEN rating = "FIVE" THEN 1 ELSE 0 END AS five_star_rating,
                CASE WHEN (rating != "FIVE" AND rating != "ZERO") THEN 1 ELSE 0 END AS one_to_four_star_rating,
                CASE
                    WHEN rating = "FIVE" THEN 5
                    WHEN rating = "FOUR" THEN 4
                    WHEN rating = "THREE" THEN 3
                    WHEN rating = "TWO" THEN 2
                    WHEN rating = "ONE" THEN 1
                    WHEN rating = "ZERO" THEN 0
                    ELSE NULL
                END AS rating_2,
                CASE WHEN rating = "FIVE" THEN "5 star reviews" ELSE "others" END AS rating_3
            FROM `the-pulse-405018.the_pulse.reviews_bq`
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Follows the same pattern as the other tables

class CallsForLSTool(BaseTool):
    name: str = "Calls_ForLS"
    description: str = "Provides information about the structure and calculations in the calls_forLS table."

    @property
    def table_info(self):
        return {
            "name": "calls_forLS",
            "description": "Provides information about call data.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "call_date": "Calculated using: PARSE_DATE('%Y-%m-%d', SUBSTR(call_started_at, 1, 10)) AS call_date",
                "Total_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" THEN 1 ELSE 0 END) AS Total_Calls",
                "Answered_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" AND call_status = \"completed\" THEN 1 ELSE 0 END) AS Answered_Calls",
                "Missed_Calls": "Calculated using: SUM(CASE WHEN call_direction = \"inbound\" AND first_time = \"True\" AND call_status = \"no-answer\" THEN 1 ELSE 0 END) AS Missed_Calls"
            },
            "source_table": "the-pulse-405018.the_pulse.calls_bq",
            "clustering": "CLUSTER BY ghl_location_id",
            "partitioning": "none",
            "grouping": "GROUP BY ghl_location_id, ghl_location_name, call_started_at",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.calls_forLS`
            CLUSTER BY ghl_location_id AS
            SELECT
                ghl_location_id,
                ghl_location_name,
                PARSE_DATE('%Y-%m-%d', SUBSTR(call_started_at, 1, 10)) AS call_date,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" THEN 1 ELSE 0 END) AS Total_Calls,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" AND call_status = "completed" THEN 1 ELSE 0 END) AS Answered_Calls,
                SUM(CASE WHEN call_direction = "inbound" AND first_time = "True" AND call_status = "no-answer" THEN 1 ELSE 0 END) AS Missed_Calls
            FROM `the-pulse-405018.the_pulse.calls_bq`
            GROUP BY ghl_location_id, ghl_location_name, call_started_at
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        elif query in ["clustering", "partitioning", "grouping", "full_query"]:
            return f"{query.capitalize()}: {self.table_info[query]}"
        elif query == "source_table":
            return f"Source Table: {self.table_info['source_table']}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Follows the same pattern as the other tables

class OpportunitiesForLSTool(BaseTool):
    name: str = "Opportunities_ForLS"
    description: str = "Provides information about the structure and calculations in the opportunities_forLS table."

    @property
    def table_info(self):
        return {
            "name": "opportunities_forLS",
            "description": "Provides information about opportunities data.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table",
                "ghl_location_name": "Directly selected from the source table",
                "contact_id": "Directly selected from the source table",
                "contact_email": "Directly selected from the source table",
                "contact_tags": "Directly selected from the source table",
                "opportunity_created_date": "Directly selected from the source table",
                "opportunity_id": "Directly selected from the source table",
                "Leads": "CASE WHEN opportunity_id != '' THEN 1 ELSE 0 END",
                "Qualified": "CASE WHEN opportunity_id != '' AND opportunity_status != 'abandoned' THEN 1 ELSE 0 END",
                "Retained": "CASE WHEN opportunity_id != '' AND opportunity_stage_name = 'Retained' THEN 1 ELSE 0 END",
                "Open_1": "CASE WHEN opportunity_id != '' AND (opportunity_status = 'open' OR opportunity_status = 'open') THEN 1 ELSE 0 END",
                "Lost": "CASE WHEN opportunity_id != '' AND opportunity_status = 'lost' THEN 1 ELSE 0 END",
                "Signed_Cases": "CASE WHEN opportunity_id != '' AND opportunity_status = 'won' THEN 1 ELSE 0 END",
                "opportunity_pipeline_name": "Directly selected from the source table",
                "opportunity_pipeline_id": "Directly selected from the source table",
                "opportunity_stage_name": "Directly selected from the source table",
                "opportunity_stage_id": "Directly selected from the source table",
                "opportunity_source": "CASE WHEN opportunity_source = '' THEN 'Not Indicated' WHEN opportunity_source IS NULL THEN 'Not Indicated' WHEN opportunity_source = 'undefined' THEN 'Undefined' WHEN opportunity_source = 'LSA' THEN 'Undefined' ELSE opportunity_source END",
                "Revenue": "SELECT opportunity_value AS Revenue",
                "Status": "SELECT opportunity_status AS Status",
                "opportunity_assigned_user": "Directly selected from the source table",
                "Location": "CASE WHEN opportunity_location = '' THEN 'Not Indicated' WHEN opportunity_location IS NULL THEN 'Not Indicated' ELSE opportunity_location END",
                "Medium": "SELECT opportunity_medium AS Medium",
                "Type": "SELECT opportunity_case_type AS Type"
            },
            "source_table": "the-pulse-405018.the_pulse.lfgm_opportunities_bq",
            "partitioning": "PARTITION BY opportunity_created_date",
            "clustering": "CLUSTER BY ghl_location_id, opportunity_source, Location, Medium",
            "grouping": "none",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.opportunities_forLS`
            PARTITION BY opportunity_created_date
            CLUSTER BY ghl_location_id, opportunity_source, Location, Medium
            AS
            SELECT
                ghl_location_id,
                ghl_location_name,
                contact_id,
                contact_email,
                contact_tags,
                opportunity_created_date,
                opportunity_id,
                CASE WHEN opportunity_id != "" THEN 1 ELSE 0 END AS Leads,
                CASE WHEN opportunity_id != "" AND opportunity_status != "abandoned" THEN 1 ELSE 0 END AS Qualified,
                CASE WHEN opportunity_id != "" AND opportunity_stage_name = "Retained" THEN 1 ELSE 0 END AS Retained,
                CASE WHEN opportunity_id != "" AND (opportunity_status = "open" OR opportunity_status = "open") THEN 1 ELSE 0 END AS Open_1, -- Not yet less retained
                CASE WHEN opportunity_id != "" AND opportunity_status = "lost" THEN 1 ELSE 0 END AS Lost,
                CASE WHEN opportunity_id != "" AND opportunity_status = "won" THEN 1 ELSE 0 END AS Signed_Cases,
                opportunity_pipeline_name,
                opportunity_pipeline_id,
                opportunity_stage_name,
                opportunity_stage_id,
                CASE
                    WHEN opportunity_source = "" THEN "Not Indicated"
                    WHEN opportunity_source IS NULL THEN "Not Indicated"
                    WHEN opportunity_source = "undefined" THEN "Undefined"
                    WHEN opportunity_source = "LSA" THEN "Undefined"
                    ELSE opportunity_source
                END AS opportunity_source,
                opportunity_value AS Revenue,
                opportunity_status AS Status,
                opportunity_assigned_user,
                CASE
                    WHEN opportunity_location = "" THEN "Not Indicated"
                    WHEN opportunity_location IS NULL THEN "Not Indicated"
                    ELSE opportunity_location
                END AS Location,
                opportunity_medium AS Medium,
                opportunity_case_type AS Type
            FROM `the-pulse-405018.the_pulse.lfgm_opportunities_bq`
            UNION ALL
            SELECT
                "ABC" AS ghl_location_id,
                "ABC" AS ghl_location_name,
                "ABC" AS contact_id,
                "ABC" AS contact_email,
                "ABC" AS contact_tags,
                NULL AS opportunity_created_date,
                "" AS opportunity_id,
                1 AS Leads,
                1 AS Qualified,
                1 AS Retained,
                1 AS Open_1,
                1 AS Lost,
                1 AS Signed_Cases,
                "" AS opportunity_pipeline_name,
                "" AS opportunity_pipeline_id,
                "" AS opportunity_stage_name,
                "Not Indicated" AS opportunity_stage_id,
                "Not Indicated" AS opportunity_source,
                0 AS Revenue,
                "" AS Status,
                "" AS opportunity_assigned_user,
                "Not Indicated" AS Location,
                "AS" AS Medium,
                "" AS Type
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        elif query in ["clustering", "partitioning", "grouping", "full_query"]:
            return f"{query.capitalize()}: {self.table_info[query]}"
        elif query == "source_table":
            return f"Source Table: {self.table_info['source_table']}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")


# Follows the same pattern as the other tables

class AdExpenseDataMonthlyForLSTool(BaseTool):
    name: str = "Ad_Expense_Data_Monthly_ForLS"
    description: str = "Provides information about the structure and calculations in the ad_expense_data_monthly_forLS table."

    @property
    def table_info(self):
        return {
            "name": "ad_expense_data_monthly_forLS",
            "description": "Provides information about the structure and calculations in the ad_expense_data_monthly_forLS table.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "formatted_expense_monthyear": "Directly selected from the source table using: SELECT formatted_expense_monthyear",
                "Month": "Calculated using: 1, 3, 6, or 12 AS Month depending on the time period",
                "ad_expense": "Calculated using: SUM(ad_expense) AS ad_expense"
            },
            "source_table": "the-pulse-405018.the_pulse.ad_expense_data_forLS",
            "time_periods": [
                "Last 1 month: WHERE DATE_TRUNC(formatted_expense_monthyear, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)",
                "Last 3 months: WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH) AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 6 months: WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH) AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 12 months: WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH) AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)"
            ],
            "grouping": "GROUP BY ghl_location_id, ghl_location_name, formatted_expense_monthyear",
            "partitioning": "none",
            "clustering": "none",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.ad_expense_data_monthly_forLS` AS
            -- For the last month
            SELECT
                ghl_location_id,
                ghl_location_name,
                formatted_expense_monthyear,
                1 AS Month, -- Last month
                SUM(ad_expense) AS ad_expense
            FROM `the-pulse-405018.the_pulse.ad_expense_data_forLS`
            WHERE DATE_TRUNC(formatted_expense_monthyear, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, formatted_expense_monthyear

            UNION ALL

            -- For the last 3 months
            SELECT
                ghl_location_id,
                ghl_location_name,
                formatted_expense_monthyear,
                3 AS Month, -- Last 3 months
                SUM(ad_expense) AS ad_expense
            FROM `the-pulse-405018.the_pulse.ad_expense_data_forLS`
            WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH)
              AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, formatted_expense_monthyear

            UNION ALL

            -- For the last 6 months
            SELECT
                ghl_location_id,
                ghl_location_name,
                formatted_expense_monthyear,
                6 AS Month, -- Last 6 months
                SUM(ad_expense) AS ad_expense
            FROM `the-pulse-405018.the_pulse.ad_expense_data_forLS`
            WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH)
              AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, formatted_expense_monthyear

            UNION ALL

            -- For the last 12 months
            SELECT
                ghl_location_id,
                ghl_location_name,
                formatted_expense_monthyear,
                12 AS Month, -- Last 12 months
                SUM(ad_expense) AS ad_expense
            FROM `the-pulse-405018.the_pulse.ad_expense_data_forLS`
            WHERE formatted_expense_monthyear >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH)
              AND formatted_expense_monthyear < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, formatted_expense_monthyear
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        elif query in ["clustering", "partitioning", "grouping", "full_query", "time_periods"]:
            return f"{query.capitalize()}: {self.table_info[query]}"
        elif query == "source_table":
            return f"Source Table: {self.table_info['source_table']}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Follows the same pattern as the other tables

class OpportunitiesMonthlyForLSTool(BaseTool):
    name: str = "Opportunities_Monthly_ForLS"
    description: str = "Provides information about the structure and calculations in the opportunities_monthly_forLS table."

    @property
    def table_info(self):
        return {
            "name": "opportunities_monthly_forLS",
            "description": "Provides information about the structure and calculations in the opportunities_monthly_forLS table.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "opportunity_created_date": "Directly selected from the source table using: SELECT opportunity_created_date",
                "Month": "Calculated using: 1, 3, 6, or 12 AS Month depending on the time period",
                "Leads": "Calculated using: SUM(Leads) AS Leads",
                "Qualified": "Calculated using: SUM(Qualified) AS Qualified",
                "Retained": "Calculated using: SUM(Retained) AS Retained",
                "Lost": "Calculated using: SUM(Lost) AS Lost",
                "Signed_Cases": "Calculated using: SUM(Signed_Cases) AS Signed_Cases",
                "Revenue": "Calculated using: SUM(Revenue) AS Revenue"
            },
            "source_table": "the-pulse-405018.the_pulse.opportunities_forLS",
            "time_periods": [
                "Last 1 month: WHERE DATE_TRUNC(opportunity_created_date, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)",
                "Last 3 months: WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH) AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 6 months: WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH) AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 12 months: WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH) AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)"
            ],
            "grouping": "GROUP BY ghl_location_id, ghl_location_name, opportunity_created_date",
            "partitioning": "none",
            "clustering": "none",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.opportunities_monthly_forLS` AS
            SELECT
                ghl_location_id,
                ghl_location_name,
                opportunity_created_date,
                1 AS Month, -- Last month
                SUM(Leads) AS Leads,
                SUM(Qualified) AS Qualified,
                SUM(Retained) AS Retained,
                SUM(Lost) AS Lost,
                SUM(Signed_Cases) AS Signed_Cases,
                SUM(Revenue) AS Revenue
            FROM `the-pulse-405018.the_pulse.opportunities_forLS`
            WHERE DATE_TRUNC(opportunity_created_date, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, opportunity_created_date

            UNION ALL

            SELECT
                ghl_location_id,
                ghl_location_name,
                opportunity_created_date,
                3 AS Month, -- Last 3 months
                SUM(Leads) AS Leads,
                SUM(Qualified) AS Qualified,
                SUM(Retained) AS Retained,
                SUM(Lost) AS Lost,
                SUM(Signed_Cases) AS Signed_Cases,
                SUM(Revenue) AS Revenue
            FROM `the-pulse-405018.the_pulse.opportunities_forLS`
            WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH)
              AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, opportunity_created_date

            UNION ALL

            SELECT
                ghl_location_id,
                ghl_location_name,
                opportunity_created_date,
                6 AS Month, -- Last 6 months
                SUM(Leads) AS Leads,
                SUM(Qualified) AS Qualified,
                SUM(Retained) AS Retained,
                SUM(Lost) AS Lost,
                SUM(Signed_Cases) AS Signed_Cases,
                SUM(Revenue) AS Revenue
            FROM `the-pulse-405018.the_pulse.opportunities_forLS`
            WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH)
              AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, opportunity_created_date

            UNION ALL

            SELECT
                ghl_location_id,
                ghl_location_name,
                opportunity_created_date,
                12 AS Month, -- Last 12 months
                SUM(Leads) AS Leads,
                SUM(Qualified) AS Qualified,
                SUM(Retained) AS Retained,
                SUM(Lost) AS Lost,
                SUM(Signed_Cases) AS Signed_Cases,
                SUM(Revenue) AS Revenue
            FROM `the-pulse-405018.the_pulse.opportunities_forLS`
            WHERE opportunity_created_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH)
              AND opportunity_created_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, opportunity_created_date
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        elif query == "time_periods":
            return f"Time Periods: {self.table_info['time_periods']}"
        elif query == "grouping":
            return f"Grouping: {self.table_info['grouping']}"
        elif query == "full_query":
            return f"Full Query: {self.table_info['full_query']}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")


# Follows the same pattern as the other tables

class CallsMonthlyForLSTool(BaseTool):
    name: str = "Calls_Monthly_ForLS"
    description: str = "Provides information about the structure and calculations in the calls_monthly_forLS table."

    @property
    def table_info(self):
        return {
            "name": "calls_monthly_forLS",
            "description": "Provides information about the structure and calculations in the calls_monthly_forLS table.",
            "fields": {
                "ghl_location_id": "Directly selected from the source table using: SELECT ghl_location_id",
                "ghl_location_name": "Directly selected from the source table using: SELECT ghl_location_name",
                "call_date": "Directly selected from the source table using: SELECT call_date",
                "Month": "Calculated using: 1, 3, 6, or 12 AS Month depending on the time period",
                "Total_Calls": "Calculated using: SUM(Total_Calls) AS Total_Calls",
                "Answered_Calls": "Calculated using: SUM(Answered_Calls) AS Answered_Calls"
            },
            "source_table": "the-pulse-405018.the_pulse.calls_forLS",
            "time_periods": [
                "Last 1 month: WHERE DATE_TRUNC(call_date, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)",
                "Last 3 months: WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH) AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 6 months: WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH) AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)",
                "Last 12 months: WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH) AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)"
            ],
            "grouping": "GROUP BY ghl_location_id, ghl_location_name, call_date",
            "partitioning": "none",
            "clustering": "none",
            "full_query": """
            CREATE OR REPLACE TABLE `the-pulse-405018.the_pulse.calls_monthly_forLS` AS
            -- For the last month
            SELECT
                ghl_location_id,
                ghl_location_name,
                call_date,
                1 AS Month, -- Last month
                SUM(Total_Calls) AS Total_Calls,
                SUM(Answered_Calls) AS Answered_Calls
            FROM `the-pulse-405018.the_pulse.calls_forLS`
            WHERE DATE_TRUNC(call_date, MONTH) = DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, call_date

            UNION ALL

            -- For the last 3 months
            SELECT
                ghl_location_id,
                ghl_location_name,
                call_date,
                3 AS Month, -- Last 3 months
                SUM(Total_Calls) AS Total_Calls,
                SUM(Answered_Calls) AS Answered_Calls
            FROM `the-pulse-405018.the_pulse.calls_forLS`
            WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH), MONTH)
              AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, call_date

            UNION ALL

            SELECT
                ghl_location_id,
                ghl_location_name,
                call_date,
                6 AS Month, -- Last 6 months
                SUM(Total_Calls) AS Total_Calls,
                SUM(Answered_Calls) AS Answered_Calls
            FROM `the-pulse-405018.the_pulse.calls_forLS`
            WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH), MONTH)
              AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, call_date

            UNION ALL

            SELECT
                ghl_location_id,
                ghl_location_name,
                call_date,
                12 AS Month, -- Last 12 months
                SUM(Total_Calls) AS Total_Calls,
                SUM(Answered_Calls) AS Answered_Calls
            FROM `the-pulse-405018.the_pulse.calls_forLS`
            WHERE call_date >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH), MONTH)
              AND call_date < DATE_TRUNC(CURRENT_DATE(), MONTH)
            GROUP BY ghl_location_id, ghl_location_name, call_date
            """
        }

    def _run(self, query: str) -> str:
        query = query.lower()
        if query == "structure":
            return str(self.table_info)
        elif query in self.table_info["fields"]:
            return f"Field '{query}': {self.table_info['fields'][query]}"
        elif query == "time_periods":
            return f"Time Periods: {self.table_info['time_periods']}"
        elif query == "grouping":
            return f"Grouping: {self.table_info['grouping']}"
        elif query == "full_query":
            return f"Full Query: {self.table_info['full_query']}"
        else:
            return f"No information found for query: {query}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

#Here the agent is being setup. It receives the openai API key and a list of all tools available. Any new tools need to be defined before this, and then added to the tool repertoire. Here is also where the LLM model is defined. This script was built with the langchain_openai package in mind, but minor tweaks can be done so that other LLMs such as Claude or Gemini are used instead. Here is also where the GPT model is chosen. Current version uses GPT-4, which currently points to gpt-4-0613. If reducing costs is a priority, GPT-4o and GPT-4o mini are cheaper, but not as potent and might struggle with complex tasks. If tasks are growing too complex for the model, consider changing to gpt-4-1106-preview or gpt-4-turbo-preview, which are more expensive but more capable.

#The max_iterations variable defines how many api requests the LLM can make to answer the question. Generally speaking, the agent thinks, then uses a tool, then thinks again and uses another tool, and each combination of thought+tool makes one API request. The current workflow is: thought-DataSchemaRoadmapTool, thought-specific table tool, thought-bigquery tool, final answer. If the agent struggles at any point, if the task at hand is more complex than a simple consult, or if the BigQuery tool meets a syntax error, another thought-tool step is added to the process.  I've limited it at 5 iterations and found it enough to answer simple questions, but if the "max iterations reached" error happens frequently, it might be worth it to bump it up to 10 or more, bearing in mind this will increase the costs. I would not recommend removing the limit entirely, because the agent can get caught on loops and end up making dozens of fruitless API requests.

def setup_agent(openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    bigquery_client = bigquery.Client(credentials=credentials)

    tools = [
        Tool(
            name="BigQuery",
            func=partial(execute_bigquery, bigquery_client),
            description="Executes SQL queries on BigQuery. Input should be a valid SQL query string."
        ),
        Tool(
            name="Data Schema Roadmap",
            func=DataSchemaRoadmapTool().run,
            description="Provides information about the data schema roadmap for Looker Studio visualizations"
        ),
        Tool.from_function(
            func=CallsForLSTool()._run,
            name="Calls_ForLS",
            description="Provides information about call data from the calls_forLS table."
        ),
        Tool.from_function(
            func=OpportunitiesForLSTool()._run,
            name="Opportunities_ForLS",
            description="Provides information about opportunity data from the opportunities_forLS table."
        ),
        Tool.from_function(
            func=AdExpenseDataForLSTool()._run,
            name="Ad_Expense_Data_ForLS",
            description="Provides information about ad expense data from the ad_expense_data_forLS table."
        ),
        Tool.from_function(
            func=ReviewsForLSTool()._run,
            name="Reviews_ForLS",
            description="Provides information about review data from the reviews_forLS table."
        ),
        Tool.from_function(
            func=AdExpenseDataMonthlyForLSTool()._run,
            name="Ad_Expense_Data_Monthly_ForLS",
            description="Provides information about monthly ad expense data from the ad_expense_data_monthly_forLS table."
        ),
        Tool.from_function(
            func=OpportunitiesMonthlyForLSTool()._run,
            name="Opportunities_Monthly_ForLS",
            description="Provides information about monthly opportunities data from the opportunities_monthly_forLS table."
        ),
        Tool.from_function(
            func=CallsMonthlyForLSTool()._run,
            name="Calls_Monthly_ForLS",
            description="Provides information about monthly call data from the calls_monthly_forLS table."
        )
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5
    )
    return agent

# This is the actual agent prompt. Any new tool added to the repertoire or any changes in workflow need to be explained to the agent here, so that it knows what to do and when to use each tool. Here is also where the step by step workflow is defined. Syntax was a big issue when building this prompt - the agent originally built a query wrapped in backticks ``, and bigquery does not support this syntax. I've repeatedly instructed the agent to not do this, with success. I would not recommend removing that part of the prompt because the syntax error caused by the backticks leads the agent into a loop where it keeps trying to send the request and receiving a syntax error in return.

def run_agent(agent, prompt):
    try:
        result = agent.run(f"""Use the following steps to answer this question about the 'the_pulse' dataset
        in the 'the-pulse-405018' project:
        {prompt}
        Steps:
        1. Check the Data Schema Roadmap tool and identify the relevant table(s) for the query. Use the command "list tables".
        2. After identifying the relevant table, select the appropriate tool from the tool roster.
        3. To use the tool, do it like this:
        - For information about a specific field: `ToolName._run("field_name")`
        - For the table structure: `ToolName._run("structure")`
        - For the full SQL query: `ToolName._run("full_query")`
        - For clustering information: `ToolName._run("clustering")`
        - For partitioning information: `ToolName._run("partitioning")`
        - For grouping information: `ToolName._run("grouping")`
        - For source table information: `ToolName._run("source_table")`
        - For time periods (if applicable): `ToolName._run("time_periods")`
        4. Identify in the tool the SQL query or queries used to create the data for the looker studio visualization.
        5. Use the Bigquery tool to execute the query as described
        6. Analyze the results for relevant information.
        7. Interpret the results and provide a natural language answer.
        Remember:
        - Always use proper BigQuery syntax and table references.
        - When formatting SQL queries, ensure that the syntax is correct and there are no extraneous characters.
        - Do NOT add "```sql" before the query or "```" after. Instead, send the query directly.
        - YOUR ACTION INPUT FOR BIG QUERY TOOL SHOULD NOT BE PRECEDED BY ```sql OR ``` AND SHOULD NOT END WITH ```. DOUBLE CHECK IT BEFORE SENDING
        - ATTEMPT TO REPLICATE THE ORIGINAL QUERY INSTEAD OF MAKING YOUR OWN. ONLY MAKE YOUR OWN IF YOU CANNOT QUERY EXACTLY AS IT IS DONE IN THE TOOL.
        - If you need to modify the original query, make sure to preserve the essential structure and only change the necessary parts to answer the specific question.
        - Always consider the partitioning, clustering, and grouping of the table when writing or modifying queries for optimal performance.
        """)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# The agent implementation is done, and this is the fully functional agent. From here onwards, everything regards the streamlit front end.
st.set_page_config(layout="centered") #Defines page layout

# Checks if user is logged in and directs to the login page if not
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        'logged_in': False,
        'messages': [],
        'openai_api_key': '',
        'agent': None
    }

#Streamlit markdown and styling. The background is a black and blue conic gradient. Generally speaking, the main CSS container in streamlit is .block-container and that's where things are by default. Comments regarding each section are given below in appropriate CSS comment syntax. Those are the general definitions that apply to both the login and the chat pages.

st.markdown("""
<style>
    /* Base styles for the whole app */
    .stApp {
        background: conic-gradient(
            from 0deg at 50% 50%,
            #000000,
            #000011,
            #000033,
            #000011,
            #000000
        );
        background-attachment: fixed;
    }

    /* Center the main content vertically and horizontally */
    .block-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding-top: 0rem;
        padding-bottom: 0rem;
        height: 100vh;
    }

    /* Style for the centered header */
    .centered-header {
        width: 100%;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }

    /* Add underline to the header */
    .centered-header h1 {
        position: relative;
        padding-bottom: 10px;
    }

    .centered-header h1::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 2px;
        background-color: #ADD8E6;
    }

    /* Create a container for login elements */
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        max-width: 400px;
    }

    /* Style the input field */
    .stTextInput input {
        background-color: rgba(0, 61, 97, 0.7);
        color: white;
        border: 1px solid #0056b3;
        border-radius: 5px;
    }

    /* Style the button */
    .stButton > button {
        width: 100%;
        background-color: #0056b3;
        color: white;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        background-color: #007bff;
    }

    /* Style for error messages */
    .stAlert {
        background-color: rgba(255, 0, 0, 0.1);
        color: #ff9999;
        border: 1px solid #ff6666;
        border-radius: 5px;
        padding: 10px;
        margin-top: 1rem;
    }
</style>

<div class="login-container">
    <div class="placeholder-for-input"></div>
    <div class="placeholder-for-button"></div>
</div>

<script>
    // Move only the input and button into our custom container
    const container = document.querySelector('.login-container');
    const input = document.querySelector('.stTextInput');
    const button = document.querySelector('.stButton');

    if (input) container.replaceChild(input, container.querySelector('.placeholder-for-input'));
    if (button) container.replaceChild(button, container.querySelector('.placeholder-for-button'));
</script>
""", unsafe_allow_html=True)

# Login here has been defined as the moment when the user inputs the openai API key. No actual login credentials are needed, which is an issue because it makes the tool generally accessible to anyone. A login security check would be welcome here. Here is also a good place to take in as input the {ghl_location} placeholder variable, which will then provide the BigQuery tool with the information necessary to limit the client's query to its own ghl_location.

def attempt_login():
    if st.session_state.openai_api_key:
        with st.spinner("Setting up the agent..."):
            st.session_state.app_state['agent'] = setup_agent(st.session_state.openai_api_key)
        st.session_state.app_state['logged_in'] = True
        st.session_state.app_state['openai_api_key'] = st.session_state.openai_api_key
        st.success("Logged in successfully!")
    else:
        st.error("Please enter an OpenAI API key")

# CSS regarding the login and chat pages. The way streamlit works is that each page is defined by a python function. This is somehow analogue to defining a page in HTML and, indeed, you can see the HTML tags inside the function. HTML and CSS should be wrapped in the st.markdown() function with appropriate syntax. Any aesthetic changes specific to these pages should be done after the function call that defines it, so that it overwrites the general streamlit-CSS definitions.

# Defining login page with the receiving API key as input functionality
def login_page():
    st.markdown("""
    <div class="centered-header">
        <h1>Dashboard Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

    st.text_input("OpenAI API Key", type="password", key="openai_api_key", on_change=attempt_login)
    if st.button("LOGIN", key="login_button"):
        attempt_login()

# Defining the chatbot page
def chat_page():
    st.title("Chat with the assistant!")
    st.markdown("""
    <style>
    .stChatMessage {
        width: 200% !important;
        margin-left: -50% !important;
        margin-right: -100% !important;
    }
    .stChatContainer {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        max-width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 80% !important;
        max-width: 800px !important;
        background-color: #0E0E2C !important;
        padding: 20px !important;
        z-index: 1000 !important;
    }
    .stChatInputContainer input {
        background-color: #1E1E3F !important;
        color: white !important;
        border: 1px solid #3E3E6F !important;
        border-radius: 20px !important;
        width: 100% !important;
    }
    .main .block-container {
        padding-bottom: 100px !important;
    }
    </style>
    """, unsafe_allow_html=True)

#Here is where the messaging part of the chatbot is defined. Chat input container is defined here. First we have a markdown defining the page structure. Bear in mind everything here is still inside the chat_page() function and consider the indenting when editing. The "if prompt" section receives the user prompt and activates the agent at "try: full_response = run_agent". Everything the agent thinks and does gets logged on the streamlit terminal, so if debugging is necessary, the terminal can provide explanations as to what went wrong.

    for message in st.session_state.app_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    prompt = st.chat_input("Ask a question about 'the_pulse' dataset")
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt:
        st.session_state.app_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    full_response = run_agent(st.session_state.app_state['agent'], prompt)
                    message_placeholder.markdown(full_response)
                    st.session_state.app_state['messages'].append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

#Initializes the streamlit app.
def main():
    if not st.session_state.app_state['logged_in']:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
