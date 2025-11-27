# GenAI Task: Build an SHL Assessment Recommendation System

## Problem Overview

Hiring managers and recruiters often struggle to find the right assessments for the roles that they are hiring for. The current system relies on keyword searches and filters, making the process time-consuming and inefficient. Your task is to build an intelligent recommendation system that simplifies this process. Given a natural language query or a job description (JD) text or an URL (containing a JD), your application should return a list of relevant SHL assessments.

You can take a look at the data sources that you are going to work with here: [SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/)

## Your Task

Design and develop a web application that:

1. Takes a given natural language query or job description text URL
2. Recommends at minimum 5 (maximum 10) most relevant "individual test solutions" from [here](/Users/siddhantgond/Desktop/Github_Modules/google_adk_kit/shl-recommender-system/data/raw/Gen_AI_Dataset.xlsx) in the tabular format. **Note:** You need to ignore "Pre-packaged Job Solutions" category from this link
3. Each recommendation needs to have at least the following attributes:
    - Assessment name
    - URL (as given in SHL's catalog)

## Datasets Given

You need to crawl the assessment data given on the [link](https://www.shl.com/solutions/products/product-catalog/). Additionally, following data is provided, which you can use to build and test your solution.

**Datasets can be found at:** link

- **Labelled Train set:** This contains a set of 10 queries labeled by humans, most relevant assessments from the catalog. This can be used to train your model/iterate over your prompts etc.
- **Unlabeled test set:** This dataset contains a set of 9 queries – on which you have generate and submit predictions.

## Submission Materials

You need to submit the following items using the form:

### 3 URLs:
    1. API end point which can be queried using a query or piece of text and returns result in JSON (API Configurations mentioned below in Appendix 2)
    2. URL of the code on GitHub which we can see (URL of the complete code - Including your experiments and evaluation)
    3. URL of the web application frontend to test the application

### Additional Documents:
- **2-page document** outlining your approach on how you solved this problem. Most importantly, document the efforts you put into optimizing the overall performance score, including initial results and how you improved them. Write this document as concisely as possible with appropriate information
- **1-csv file** with 2 columns: `query` and `predictions` (Format given in Appendix 3). The csv file should contain the predictions on the given unlabeled test set.

### Important Notes:

These points are vital for us to evaluate the technical completeness and robustness of your solution:

- Ensure that the API URL providing access to your recommendation engine is functional, your code is accessible via a public or private (shared with us) GitHub repository, and the csv file is in the correct format
- A .csv file in the prescribed format runs over the given test set (Format given in Appendix 3)

## Evaluation Criteria

We will be using the following criteria to evaluate your solution:

### Solution Approach

- **Methodology:** The overall methodology and strategy employed to address the problem statement.
- **Data Pipeline:** The implementation of your data pipeline, including data crawling, representation, storage mechanisms, and search/retrieval functionality.
- **Technology Stack:** The extent to which modern and emerging Large Language Model (LLM) technologies and stacks were utilized.
- **Evaluation & Tracing:** The application of evaluation (evals) and tracing methodologies. We are particularly interested in the implementation and insights derived from these processes.

### Performance and Relevance

- **Recommendation Accuracy:** The performance of assessment recommendation, measured by the Mean Recall@10 against the provided test set.
- **Recommendation Balance:** The relevance and balance of recommended assessments based on query requirements.
  - **Requirement:** The system must intelligently balance recommendations when a query spans multiple domains. For instance, if a query pertains to both behavioral and technical skills, the results should contain a balanced mix of assessments.
  - **Example Scenario:**
     - **Sample Query:** "Need a Java developer who is good in collaborating with external teams and stakeholders."
     - **Expected Outcome:** The recommendation results should include a balanced set of assessments corresponding to both "Knowledge & Skills" (Test Type K) and "Personality & Behavior" (Test Type P) from the catalogue. (Test Type is present in the catalogue)



## Index: Metrics to compute accuracy

Your solution will be assessed using the following **ranking evaluation metrics**:

### 1. Mean Recall@K

This metric measures how many of the **relevant assessments** were retrieved in the **top K recommendations**, averaged across all test queries.

```
        Number of relevant assessments in top K
Recall@K = ------------------------------------
         Total relevant assessments for the query
```

```
              N
1          ∑ Recall@Kᵢ
MeanRecall@K = ----- 
              N
             i=1
```

Where N is the total number of test queries.

---

## Appendix 1: Sample Queries

Here are some of the queries that you can use to test your application.

* I am hiring for Java developers who can also collaborate effectively with my business teams.
* Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.
* Here is a JD text, can you recommend some assessment that can help me screen applications. I am hiring for an analyst and wants applications to screen using Cognitive and personality tests.

---

## Appendix 2: API Structure & Endpoints

This section outlines the required API configuration for the take-home assignment. Your API must implement the endpoints described below exactly as specified to ensure proper evaluation of your submission.

### Base requirements:

* Your API should be accessible via HTTP/HTTPS
* All responses should use proper HTTP status codes
* All data exchanges must be in JSON format

### Required Endpoints:

#### 1. Health Check Endpoint

This endpoint provides a simple status check to verify the API is running.

**Request**
• Method: GET
• Path: `<YOUR-BASE-URL>/health`

**Response**

* Content-Type: application/json
* Status Code: 200 OK (if healthy)
* Body:

```json
{
  "status": "healthy"
}
```

#### 2. Assessment Recommendation Endpoint

This endpoint accepts a job description or Natural language query and returns recommended relevant assessments (At most 10, minimum 1) based on the input.

**Request**
• Method: POST
• Path: `<YOUR-BASE-URL>/recommend`
• Content-Type: application/json
• Body:

```json
{
  "query": "JD/query in string"
}
```

**Response**

* Content-Type: application/json
* Status Code: 200 OK (if successful)
* Body:

```json
{
    "recommended_assessments": [
        {
            "url": "Valid URL in string",
            "adaptive_support": "Yes/No",
            "description": "Description in string",
            "duration": 60,
            "remote_support": "Yes/No",
            "test_type": ["List of string"]
        }
    ]
}
```

---

## Response Fields Explanation (for `/recommend` endpoint response)

| Field            | Type             | Description                                                                 |
| ---------------- | ---------------- | --------------------------------------------------------------------------- |
| url              | String           | Valid URL to the assessment resource                                        |
| name             | String           | Name of the assessment                                                      |
| adaptive_support | String           | Either “Yes” or “No” indicating if the assessment supports adaptive testing |
| description      | String           | Detailed description of the assessment                                      |
| duration         | Integer          | Duration of the assessment in minutes                                       |
| remote_support   | String           | Either “Yes” or “No” indicating if the assessment can be taken remotely     |
| test_type        | Array of Strings | Categories or types of the assessment                                       |

---

## Example Response

```json
{
   "recommended_assessments": [
      {
         "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
         "name": "Python (New)",
         "adaptive_support": "No",
         "description": "Multi-choice test that measures the knowledge of Python programming, databases, modules and library. For...",
         "duration": 11,
         "remote_support": "Yes",
         "test_type": ["Knowledge & Skills"]
      },
      {
         "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/",
         "name": "Technology Professional 8.0 Job Focused Assessment",
         "adaptive_support": "No",
         "description": "The Technology Job Focused Assessment assesses key behavioral attributes required for success in fast-paced roles.",
         "duration": 16,
         "remote_support": "Yes",
         "test_type": ["Competencies", "Personality & Behaviour"]
      }
   ]
}
```

**Testing your API**: Before submitting your API link, please verify that:

1. Both endpoints are functioning correctly
2. The response formats match exactly what is given

---

## Appendix 3: Submission Data format

We have an automated pipeline that will evaluate your results so make sure you follow the format as given.

The CSV file should be submitted in the format below.

| Query   | Assessment_url         |
| ------- | ---------------------- |
| Query 1 | Recommendation 1 (URL) |
| Query 1 | Recommendation 2 (URL) |
| Query 1 | Recommendation 3 (URL) |
| ...     | ...                    |
| Query 2 | Recommendation 1       |

Note: the submission should be in exactly the above format, if the above format is not followed then you will not be scored.

---


