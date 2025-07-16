#!/usr/bin/env python3
"""
Enhanced PDF text extraction for legal documents.
Handles various PDF formats and extracts structured legal information.
"""

import PyPDF2
import re
import os
from typing import Dict, List, Optional, Tuple

class EnhancedPDFExtractor:
    """Enhanced PDF extractor for legal documents with better text parsing."""
    
    def __init__(self):
        self.legal_patterns = {
            'case_name': r'(.+?)\s+v\.\s+(.+?)(?:\s*\(|$)',
            'citation': r'(\d{4})\s+AIR\s+(\d+)',
            'court': r'SUPREME\s+COURT|HIGH\s+COURT',
            'date': r'(\d{1,2})\s+(\w+)\s+(\d{4})',
            'article': r'Article\s+(\d+)',
            'section': r'Section\s+(\d+)',
            'act': r'(.+?)\s+Act,\s+(\d{4})'
        }
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 5) -> str:
        """Extract text from PDF with error handling."""
        try:
            if not os.path.exists(pdf_path):
                return ""
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Try to extract text from pages
                for page_num in range(min(max_pages, len(reader.pages))):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
                
                return text
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_case_metadata(self, text: str) -> Dict[str, str]:
        """Extract case metadata from text."""
        metadata = {
            'case_name': '',
            'citation': '',
            'court': '',
            'date': '',
            'parties': {'petitioner': '', 'respondent': ''},
            'articles': [],
            'sections': [],
            'acts': []
        }
        
        # Extract case name
        case_match = re.search(self.legal_patterns['case_name'], text, re.IGNORECASE)
        if case_match:
            metadata['case_name'] = f"{case_match.group(1).strip()} vs {case_match.group(2).strip()}"
            metadata['parties']['petitioner'] = case_match.group(1).strip()
            metadata['parties']['respondent'] = case_match.group(2).strip()
        
        # Extract citation
        citation_match = re.search(self.legal_patterns['citation'], text)
        if citation_match:
            metadata['citation'] = f"{citation_match.group(1)} AIR {citation_match.group(2)}"
        
        # Extract court
        court_match = re.search(self.legal_patterns['court'], text, re.IGNORECASE)
        if court_match:
            metadata['court'] = court_match.group(0)
        
        # Extract articles
        articles = re.findall(self.legal_patterns['article'], text, re.IGNORECASE)
        metadata['articles'] = list(set(articles))
        
        # Extract sections
        sections = re.findall(self.legal_patterns['section'], text, re.IGNORECASE)
        metadata['sections'] = list(set(sections))
        
        # Extract acts
        acts = re.findall(self.legal_patterns['act'], text, re.IGNORECASE)
        metadata['acts'] = list(set([act[0].strip() for act in acts]))
        
        return metadata
    
    def extract_headnote(self, text: str) -> str:
        """Extract headnote from legal document text."""
        lines = text.split('\n')
        headnote_lines = []
        in_headnote = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Look for headnote markers
            if 'headnote' in line_lower or 'head note' in line_lower:
                in_headnote = True
                continue
            
            # Look for end of headnote
            if in_headnote and any(marker in line_lower for marker in ['judgment', 'appeal', 'petitioner', 'respondent']):
                if len(line.strip()) > 20:  # Meaningful content
                    break
            
            if in_headnote and line.strip():
                headnote_lines.append(line.strip())
        
        # If no explicit headnote, extract first meaningful paragraph
        if not headnote_lines:
            for line in lines:
                if line.strip() and len(line.strip()) > 50:
                    headnote_lines.append(line.strip())
                    break
        
        return ' '.join(headnote_lines[:10])  # Limit to first 10 lines
    
    def extract_judgment_summary(self, text: str) -> str:
        """Extract judgment summary from text."""
        lines = text.split('\n')
        summary_lines = []
        in_summary = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Look for judgment section
            if any(keyword in line_lower for keyword in ['judgment', 'held', 'decision', 'conclusion']):
                in_summary = True
                continue
            
            if in_summary and line.strip() and len(line.strip()) > 20:
                summary_lines.append(line.strip())
                
                # Limit summary length
                if len(summary_lines) >= 5:
                    break
        
        return ' '.join(summary_lines)
    
    def analyze_legal_document(self, pdf_path: str) -> Dict[str, any]:
        """Complete analysis of a legal document."""
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return {
                'error': 'Could not extract text from PDF',
                'file_path': pdf_path
            }
        
        metadata = self.extract_case_metadata(text)
        headnote = self.extract_headnote(text)
        summary = self.extract_judgment_summary(text)
        
        return {
            'file_path': pdf_path,
            'metadata': metadata,
            'headnote': headnote,
            'summary': summary,
            'text_length': len(text),
            'extracted_text': text[:1000]  # First 1000 chars for preview
        }

# Global instance
extractor = EnhancedPDFExtractor()

def analyze_case_1116():
    """Specific analysis for Makhan Singh case."""
    pdf_path = 'judgements_pdfs/-0___jonew__judis__1116.pdf'
    return extractor.analyze_legal_document(pdf_path)

if __name__ == "__main__":
    # Test the enhanced extractor
    result = analyze_case_1116()
    
    print("=" * 80)
    print("ENHANCED LEGAL DOCUMENT ANALYSIS")
    print("=" * 80)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"File: {result['file_path']}")
        print(f"Text Length: {result['text_length']}")
        print()
        
        print("METADATA:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        print()
        
        print("HEADNOTE:")
        print(result['headnote'][:500] + "...")
        print()
        
        print("SUMMARY:")
        print(result['summary'][:500] + "...")