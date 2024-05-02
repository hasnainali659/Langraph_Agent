import re
import json


def extract_summary(text):
    """
    Extracts the detailed summary and important summary parts from the given text.

    Args:
        text (str): The input text from which to extract the summaries.

    Returns:
        dict: A dictionary containing the extracted summaries.
            The dictionary has the following keys:
            - 'detailed_summary': The detailed summary extracted from the text.
            - 'important_summary': The important summary extracted from the text,
              with additional details extracted using the `extract_summary_details` function.
    """
    detailed_summary_pattern = r"Detailed Summary:(.*?)Important Part Summary:"
    important_summary_pattern = r"Important Part Summary:(.*?)$"

    detailed_summary_match = re.search(detailed_summary_pattern, text, re.DOTALL)
    important_summary_match = re.search(important_summary_pattern, text, re.DOTALL)

    detailed_summary = (
        detailed_summary_match.group(1).strip() if detailed_summary_match else ""
    )
    important_summary = (
        important_summary_match.group(1).strip() if important_summary_match else ""
    )

    important_summary_details = extract_summary_details(important_summary)

    return {
        "detailed_summary": detailed_summary,
        "important_summary": important_summary_details,
    }


def extract_summary_details(summary_text):
    """
    Extracts summary details from the given summary text.

    Args:
        summary_text (str): The text containing summary details.

    Returns:
        list: A list of dictionaries containing the extracted summary details.
              Each dictionary has the following keys: 'page_line', 'topic', 'summary'.
    """

    summary_details = []
    pattern = r"Page Line: (\d+-\d+)\nTopic: (.*?)\nSummary: (.*?)\n\n"
    matches = re.findall(pattern, summary_text)

    for match in matches:
        page_line = match[0]
        topic = match[1]
        summary = match[2]
        summary_details.append(
            {"page_line": page_line, "topic": topic, "summary": summary}
        )

    return summary_details
