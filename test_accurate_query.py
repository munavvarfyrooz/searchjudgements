#!/usr/bin/env python3
"""
Test script for accurate legal case queries.
This demonstrates the corrected system for Makhan Singh case.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accurate_case_query import accurate_query

def test_makhan_singh_case():
    """Test the accurate query system for Makhan Singh case."""
    
    print("=" * 80)
    print("TESTING ACCURATE LEGAL CASE QUERY SYSTEM")
    print("=" * 80)
    print()
    
    # Test queries
    test_queries = [
        "Makhan Singh case headnote",
        "What is the constitutional issue in Makhan Singh vs State of Punjab?",
        "1964 AIR 381 headnote",
        "Is Makhan Singh case about emergency powers?",
        "Case 1116 summary"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        response, sources = accurate_query.handle_query(query)
        
        print("Response:")
        print(response)
        print()
        print(f"Sources: {sources}")
        print("=" * 80)
        print()

if __name__ == "__main__":
    test_makhan_singh_case()