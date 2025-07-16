#!/usr/bin/env python3
"""
Simple wrapper for accurate case queries using the FinalAccurateQuery system.
"""

from final_accurate_query import FinalAccurateQuery

# Create global instance
_query_system = FinalAccurateQuery()

def accurate_case_query(query: str) -> str:
    """
    Get accurate case information based on verified PDF content.
    
    Args:
        query: The query string about a legal case
        
    Returns:
        Accurate response based on verified document content
    """
    response, sources = _query_system.handle_query(query)
    return response

# Test function
if __name__ == "__main__":
    test_queries = [
        "Makhan Singh case headnote",
        "What is the constitutional issue in Makhan Singh vs State of Punjab?",
        "Is Makhan Singh case about emergency powers or criminal conviction?",
        "1964 AIR 381 summary"
    ]
    
    print("=== ACCURATE CASE QUERY TESTS ===\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        response = accurate_case_query(query)
        print(response)
        print("=" * 70)
        print()