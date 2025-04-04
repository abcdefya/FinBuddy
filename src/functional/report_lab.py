"""
Financial Report Generator

This module provides utilities for generating professional financial research reports
using ReportLab. It creates structured PDF documents with financial data, charts,
and analysis sections suitable for investment research.
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Annotated, Dict, Any, Optional, Union, List, Tuple

from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.platypus import (
    SimpleDocTemplate,
    Frame,
    Paragraph,
    Image,
    PageTemplate,
    FrameBreak,
    Spacer,
    Table,
    TableStyle,
    NextPageTemplate,
    PageBreak,
)
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

from ..data import FMPUtils, YFinanceUtils
from .analyzer import ReportAnalysisUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportLabUtils:
    """
    Financial report generation utilities using ReportLab.
    
    Provides methods for creating professional equity research reports with
    company analysis, financial metrics, and performance visualizations.
    """

    @staticmethod
    def build_annual_report(
        ticker_symbol: Annotated[str, "ticker symbol"],
        save_path: Annotated[str, "path to save the annual report pdf"],
        operating_results: Annotated[
            str,
            "a paragraph of text: the company's income summarization from its financial report",
        ],
        market_position: Annotated[
            str,
            "a paragraph of text: the company's current situation and end market (geography), major customers (blue chip or not), market share from its financial report",
        ],
        business_overview: Annotated[
            str,
            "a paragraph of text: the company's description and business highlights from its financial report",
        ],
        risk_assessment: Annotated[
            str,
            "a paragraph of text: the company's risk assessment from its financial report",
        ],
        competitors_analysis: Annotated[
            str,
            "a paragraph of text: the company's competitors analysis from its financial report and competitors' financial report",
        ],
        share_performance_image_path: Annotated[
            str, "path to the share performance image"
        ],
        pe_eps_performance_image_path: Annotated[
            str, "path to the PE and EPS performance image"
        ],
        filing_date: Annotated[str, "filing date of the analyzed financial report"],
    ) -> str:
        """
        Generate a comprehensive equity research report in PDF format.
        
        Creates a professionally formatted report including company overview,
        financial metrics, market position, risk assessment, competitive analysis,
        and performance charts.
        
        Args:
            ticker_symbol: Stock ticker symbol
            save_path: Path to save the generated PDF
            operating_results: Text analysis of income and operations
            market_position: Text analysis of market standing
            business_overview: Text description of company and business
            risk_assessment: Text analysis of key risks
            competitors_analysis: Text analysis of competitive landscape
            share_performance_image_path: Path to stock price chart image
            pe_eps_performance_image_path: Path to PE/EPS chart image
            filing_date: Date of the financial report being analyzed
            
        Returns:
            Success message or error traceback
        """
        try:
            # Configure page layout
            page_width, page_height = pagesizes.A4
            left_column_width = page_width * 2 / 3
            right_column_width = page_width - left_column_width
            margin = 4

            # Prepare output file path
            pdf_path = ReportLabUtils._prepare_output_path(save_path, ticker_symbol)
            
            # Create document with A4 page size
            doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)
            
            # Define page templates for different layouts
            templates = ReportLabUtils._create_page_templates(page_width, page_height, margin, left_column_width)
            doc.addPageTemplates(templates)
            
            # Define text and table styles
            styles = ReportLabUtils._create_styles()
            
            # Get company name and currency
            company_info = YFinanceUtils.get_stock_info(ticker_symbol)
            company_name = company_info.get("shortName", ticker_symbol)
            currency = company_info.get("currency", "USD")
            
            # Create document content
            content = []
            
            # Build first page with two-column layout
            ReportLabUtils._add_first_page_content(
                content, 
                ticker_symbol,
                company_name,
                business_overview, 
                market_position, 
                operating_results,
                currency,
                styles, 
                left_column_width, 
                right_column_width,
                margin,
                filing_date,
                share_performance_image_path,
                pe_eps_performance_image_path
            )
            
            # Add second page with single-column layout
            ReportLabUtils._add_second_page_content(
                content,
                risk_assessment,
                competitors_analysis,
                styles
            )
            
            # Build the PDF document
            doc.build(content)
            
            logger.info(f"Successfully generated annual report for {ticker_symbol}")
            return "Annual report generated successfully."
            
        except Exception as e:
            logger.error(f"Failed to generate report for {ticker_symbol}: {str(e)}")
            return traceback.format_exc()

    @staticmethod
    def _prepare_output_path(save_path: str, ticker_symbol: str) -> str:
        """
        Prepare the output file path for the PDF report.
        
        Args:
            save_path: Base path for saving the file
            ticker_symbol: Stock ticker symbol
            
        Returns:
            Fully qualified path for the PDF file
        """
        if os.path.isdir(save_path):
            pdf_path = os.path.join(save_path, f"{ticker_symbol}_Equity_Research_report.pdf")
        else:
            pdf_path = save_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        return pdf_path

    @staticmethod
    def _create_page_templates(
        page_width: float, 
        page_height: float, 
        margin: float, 
        left_column_width: float
    ) -> List[PageTemplate]:
        """
        Create page templates for different report layouts.
        
        Args:
            page_width: Width of the page
            page_height: Height of the page
            margin: Page margin size
            left_column_width: Width of the left column
            
        Returns:
            List of PageTemplate objects
        """
        # Calculate dimensions
        right_column_width = page_width - left_column_width
        
        # First page: Two column layout (2/3 + 1/3)
        frame_left = Frame(
            margin,
            margin,
            left_column_width - margin * 2,
            page_height - margin * 2,
            id="left",
        )
        
        frame_right = Frame(
            left_column_width,
            margin,
            right_column_width - margin * 2,
            page_height - margin * 2,
            id="right",
        )
        
        # Second page: Single column layout
        single_frame = Frame(
            margin,
            margin,
            page_width - 2 * margin,
            page_height - 2 * margin,
            id="single",
        )
        
        # Third page: Two equal columns
        equal_column_width = (page_width - margin * 3) // 2
        frame_left_p2 = Frame(
            margin,
            margin,
            equal_column_width - margin * 2,
            page_height - margin * 2,
            id="left_p2",
        )
        
        frame_right_p2 = Frame(
            equal_column_width + margin,
            margin,
            equal_column_width - margin * 2,
            page_height - margin * 2,
            id="right_p2",
        )
        
        # Create page templates
        two_column = PageTemplate(id="TwoColumns", frames=[frame_left, frame_right])
        single_column = PageTemplate(id="OneCol", frames=[single_frame])
        equal_columns = PageTemplate(id="TwoColumns_p2", frames=[frame_left_p2, frame_right_p2])
        
        return [two_column, single_column, equal_columns]

    @staticmethod
    def _create_styles() -> Dict[str, Any]:
        """
        Create text and table styles for the report.
        
        Returns:
            Dictionary of style objects
        """
        styles = getSampleStyleSheet()
        
        # Main content style
        custom_style = ParagraphStyle(
            name="Custom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            alignment=TA_JUSTIFY,
        )
        
        # Title style
        title_style = ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_LEFT,
            spaceAfter=10,
        )
        
        # Section header style
        subtitle_style = ParagraphStyle(
            name="Subtitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
        
        # Main table style
        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 12),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, 0), 2, colors.black),
        ])
        
        # Financial metrics table style
        financial_table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("LINEBELOW", (0, 0), (-1, 0), 2, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
        ])
        
        return {
            "custom": custom_style,
            "title": title_style,
            "subtitle": subtitle_style,
            "table": table_style,
            "financial_table": financial_table_style
        }

    @staticmethod
    def _add_first_page_content(
        content: List, 
        ticker_symbol: str,
        company_name: str,
        business_overview: str, 
        market_position: str, 
        operating_results: str,
        currency: str,
        styles: Dict[str, Any],
        left_column_width: float,
        right_column_width: float,
        margin: float,
        filing_date: str,
        share_performance_image_path: str,
        pe_eps_performance_image_path: str
    ) -> None:
        """
        Add content to the first page of the report.
        
        Args:
            content: Content list to append to
            ticker_symbol: Stock ticker symbol
            company_name: Company name
            business_overview: Business overview text
            market_position: Market position text
            operating_results: Operating results text
            currency: Currency symbol
            styles: Dictionary of styles
            left_column_width: Width of left column
            right_column_width: Width of right column
            margin: Page margin
            filing_date: Report filing date
            share_performance_image_path: Path to share price chart
            pe_eps_performance_image_path: Path to PE/EPS chart
        """
        # Add title
        content.append(
            Paragraph(
                f"Equity Research Report: {company_name}",
                styles["title"],
            )
        )
        
        # Add business overview section
        content.append(Paragraph("Business Overview", styles["subtitle"]))
        content.append(Paragraph(business_overview, styles["custom"]))
        
        # Add market position section
        content.append(Paragraph("Market Position", styles["subtitle"]))
        content.append(Paragraph(market_position, styles["custom"]))
        
        # Add operating results section
        content.append(Paragraph("Operating Results", styles["subtitle"]))
        content.append(Paragraph(operating_results, styles["custom"]))
        
        # Add financial metrics table
        df = FMPUtils.get_financial_metrics(ticker_symbol, years=5)
        df.reset_index(inplace=True)
        df.rename(columns={"index": f"FY ({currency} mn)"}, inplace=True)
        
        table_data = [["Financial Metrics"]]
        table_data += [df.columns.to_list()] + df.values.tolist()
        
        col_widths = [(left_column_width - margin * 4) / df.shape[1]] * df.shape[1]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(styles["financial_table"])
        content.append(table)
        
        # Move to right column
        content.append(FrameBreak())
        
        # Add company info and report date
        full_length = right_column_width - 2 * margin
        company_data = [
            ["FinBuddy"],
            ["https://www.facebook.com/theanh02/"],
            ["https://github.com/abcdefya/FinBuddy/tree/main"],
            [f"Report date: {filing_date}"],
        ]
        
        company_table = Table(company_data, colWidths=[full_length])
        company_table.setStyle(styles["table"])
        content.append(company_table)
        
        # Add spacing
        content.append(Spacer(1, 0.15 * inch))
        
        # Add key financial data
        key_data = ReportAnalysisUtils.get_key_data(ticker_symbol, filing_date)
        key_data_table = [["Key data", ""]]
        key_data_table += [[k, v] for k, v in key_data.items()]
        
        col_widths = [full_length // 3 * 2, full_length // 3]
        key_table = Table(key_data_table, colWidths=col_widths)
        key_table.setStyle(styles["table"])
        content.append(key_table)
        
        # Add share performance chart
        content.append(ReportLabUtils._add_chart_section(
            "Share Performance", 
            share_performance_image_path, 
            right_column_width,
            full_length, 
            styles["table"]
        ))
        
        # Add PE & EPS performance chart
        content.append(ReportLabUtils._add_chart_section(
            "PE & EPS", 
            pe_eps_performance_image_path, 
            right_column_width,
            full_length, 
            styles["table"]
        ))

    @staticmethod
    def _add_chart_section(
        title: str, 
        image_path: str, 
        width: float,
        table_width: float,
        table_style: TableStyle
    ) -> List:
        """
        Create a chart section with title and image.
        
        Args:
            title: Chart section title
            image_path: Path to chart image
            width: Width for the image
            table_width: Width for the title table
            table_style: Style for the title table
            
        Returns:
            List of elements (table and image)
        """
        elements = []
        
        # Create title table
        data = [[title]]
        table = Table(data, colWidths=[table_width])
        table.setStyle(table_style)
        elements.append(table)
        
        # Add image
        height = width // 2
        elements.append(Image(image_path, width=width, height=height))
        
        return elements

    @staticmethod
    def _add_second_page_content(
        content: List,
        risk_assessment: str,
        competitors_analysis: str,
        styles: Dict[str, Any]
    ) -> None:
        """
        Add content to the second page of the report.
        
        Args:
            content: Content list to append to
            risk_assessment: Risk assessment text
            competitors_analysis: Competitors analysis text
            styles: Dictionary of styles
        """
        # Switch to single column layout for the second page
        content.append(NextPageTemplate("OneCol"))
        content.append(PageBreak())
        
        # Add risk assessment section
        content.append(Paragraph("Risk Assessment", styles["subtitle"]))
        content.append(Paragraph(risk_assessment, styles["custom"]))
        
        # Add competitors analysis section
        content.append(Paragraph("Competitors Analysis", styles["subtitle"]))
        content.append(Paragraph(competitors_analysis, styles["custom"]))


# Module execution entry point for testing
# if __name__ == "__main__":
#     # Test parameters
#     from datetime import datetime
    
#     TEST_TICKER = "AAPL"
#     TEST_SAVE_PATH = "test_reports"
#     TEST_DATE = datetime.now().strftime("%Y-%m-%d")
    
#     # Sample text for testing
#     SAMPLE_TEXT = """
#     This is a sample paragraph for testing the report generation functionality.
#     It contains multiple sentences to simulate a realistic analysis section.
#     The text should flow naturally across multiple lines in the final PDF document.
#     """
    
#     # Create test image paths
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Create test directory
#     os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    
#     # Generate sample charts for testing
#     def create_test_chart(filename, title):
#         plt.figure(figsize=(10, 5))
#         x = np.linspace(0, 10, 100)
#         plt.plot(x, np.sin(x))
#         plt.title(title)
#         plt.grid(True)
#         save_path = os.path.join(TEST_SAVE_PATH, filename)
#         plt.savefig(save_path)
#         plt.close()
#         return save_path
    
#     # Create test charts
#     share_chart = create_test_chart("share_performance.png", "Share Performance")
#     pe_chart = create_test_chart("pe_eps_performance.png", "PE & EPS Performance")
    
#     # Generate test report
#     try:
#         result = ReportLabUtils.build_annual_report(
#             ticker_symbol=TEST_TICKER,
#             save_path=TEST_SAVE_PATH,
#             operating_results=SAMPLE_TEXT,
#             market_position=SAMPLE_TEXT,
#             business_overview=SAMPLE_TEXT,
#             risk_assessment=SAMPLE_TEXT,
#             competitors_analysis=SAMPLE_TEXT,
#             share_performance_image_path=share_chart,
#             pe_eps_performance_image_path=pe_chart,
#             filing_date=TEST_DATE,
#         )
#         print(result)
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
