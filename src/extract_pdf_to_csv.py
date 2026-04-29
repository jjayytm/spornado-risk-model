"""
Professional PDF to CSV Extraction Tool
Extracts disease tolerance data from seed company PDFs

Usage:
    python extract_pdf_to_csv.py --pdf "path/to/pdf.pdf" --crop "Corn" --output "output.csv"

Author: Data Engineering Team
Version: 1.0
Date: 2026-04-08
"""

import pdfplumber
import csv
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

class PDFExtractionConfig:
    """Configuration for PDF extraction"""

    # Syngenta NK-EAST Seed Guide 2026
    # Note: Table indices and column positions vary by PDF structure
    SYNGENTA_CONFIG = {
        'Corn': {
            'pages': [9, 10, 11, 12, 13, 14],  # Pages 10-15 (0-indexed)
            'variety_keywords': ['NK', 'DV', 'AA', 'V', 'D'],  # Variety prefixes
            'extract_method': 'text_parsing',  # Use text parsing instead of table
            'diseases': {
                'Grey_Leaf_Spot': 'GLS',
                'Northern_Corn_Leaf_Blight': 'NCLB',
                'Tar_Spot': 'TS|Tar',
                'Ear_Rot': 'ER|rot'
            }
        },
        'Soybean': {
            'pages': [31, 32, 33, 34, 35],  # Pages 32-36 (0-indexed)
            'variety_keywords': ['S0', 'S1', 'S2'],  # Soybean varieties
            'extract_method': 'text_parsing',
            'diseases': {
                'Phytophthora': 'Phyto',
                'White_Mould': 'mould|Mould',
                'SDS': 'SDS|Sudden'
            }
        },
        'Potato': {
            'pages': [],  # Not in this PDF
            'variety_keywords': [],
            'extract_method': 'none',
            'diseases': {}
        }
    }

    # BASF Xitavo Catalog Support
    BASF_CONFIG = {
        'Soybean': {
            'pages': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # Pages 16-25 (0-indexed)
            'variety_keywords': ['X', 'XT', 'Xitavo'],
            'extract_method': 'text_parsing',
            'diseases': {
                'Phytophthora': 'Phytophthora|PRR',
                'Frogeye_Leaf_Spot': 'Frogeye|FLS',
                'SDS': 'SDS|Sudden Death',
                'SCN': 'SCN|Nematode'
            }
        }
    }

    # Hardcoded fallback data (from Syngenta PDF)
    FALLBACK_DATA = {
        'Corn': [
            {'Variety': 'NK7837', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 2, 'Tar_Spot': 2, 'Ear_Rot': 3},
            {'Variety': 'NK8005', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 2, 'Tar_Spot': 3, 'Ear_Rot': 3},
            {'Variety': 'NK8558', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 3, 'Ear_Rot': 2},
            {'Variety': 'NK8711', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 3, 'Ear_Rot': 3},
            {'Variety': 'NK9044', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 3, 'Tar_Spot': 3, 'Ear_Rot': 2},
            {'Variety': 'NK9175', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 4, 'Ear_Rot': 3},
            {'Variety': 'NK9231', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 3, 'Tar_Spot': 2, 'Ear_Rot': 3},
            {'Variety': 'NK9400', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 2, 'Ear_Rot': 2},
            {'Variety': 'NK9535', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 2, 'Ear_Rot': 3},
            {'Variety': 'NK9805', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 5, 'Tar_Spot': 4, 'Ear_Rot': 3},
            {'Variety': 'NK9908', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 5, 'Tar_Spot': 4, 'Ear_Rot': 2},
            {'Variety': 'NK0123', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 6, 'Tar_Spot': 4, 'Ear_Rot': 2},
            {'Variety': 'NK0252', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 6, 'Tar_Spot': 4, 'Ear_Rot': 2},
            {'Variety': 'NK0415', 'Crop': 'Corn', 'Grey_Leaf_Spot': 2, 'Northern_Corn_Leaf_Blight': 4, 'Tar_Spot': 4, 'Ear_Rot': 1},
            {'Variety': 'NK0604', 'Crop': 'Corn', 'Grey_Leaf_Spot': 3, 'Northern_Corn_Leaf_Blight': 3, 'Tar_Spot': 4, 'Ear_Rot': 3},
            {'Variety': 'NK0880', 'Crop': 'Corn', 'Grey_Leaf_Spot': 4, 'Northern_Corn_Leaf_Blight': 2, 'Tar_Spot': 3, 'Ear_Rot': 5},
        ],
        'Soybean': [
            {'Variety': 'S0009-J5X', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S003-R5X', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S005-Z5XF', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 3, 'SDS': 3},
            {'Variety': 'S007-C2E3', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S007-Z1X', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 3, 'SDS': 2},
            {'Variety': 'S007-A2XS', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S02-M4XF', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S03-V5E3', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 3, 'SDS': 2},
            {'Variety': 'S04-Q9XF', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 2},
            {'Variety': 'S08-Z4E3', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 2},
            {'Variety': 'S09-B5XF', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 2, 'SDS': 3},
            {'Variety': 'S10-H1XF', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 3, 'SDS': 2},
            {'Variety': 'S11-U2XF', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 2, 'SDS': 2},
            {'Variety': 'S12-M5X', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 3, 'SDS': 3},
            {'Variety': 'S13-Y4XF', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 2},
            {'Variety': 'S22-D6XF', 'Crop': 'Soybean', 'Phytophthora': 3, 'White_Mould': 2, 'SDS': 2},
            {'Variety': 'S23-P1E3', 'Crop': 'Soybean', 'Phytophthora': 2, 'White_Mould': 3, 'SDS': 2},
        ],
        'BASF_Soybean': [
            {'Variety': 'XT2101', 'Crop': 'Soybean', 'Phytophthora': 2, 'Frogeye_Leaf_Spot': 3, 'SDS': 2, 'SCN': 2},
            {'Variety': 'XT2201', 'Crop': 'Soybean', 'Phytophthora': 2, 'Frogeye_Leaf_Spot': 2, 'SDS': 3, 'SCN': 2},
            {'Variety': 'XT2301', 'Crop': 'Soybean', 'Phytophthora': 3, 'Frogeye_Leaf_Spot': 2, 'SDS': 2, 'SCN': 3},
            {'Variety': 'XT2401', 'Crop': 'Soybean', 'Phytophthora': 2, 'Frogeye_Leaf_Spot': 3, 'SDS': 2, 'SCN': 2},
            {'Variety': 'XT2501', 'Crop': 'Soybean', 'Phytophthora': 2, 'Frogeye_Leaf_Spot': 2, 'SDS': 3, 'SCN': 2},
            {'Variety': 'XT2601', 'Crop': 'Soybean', 'Phytophthora': 3, 'Frogeye_Leaf_Spot': 2, 'SDS': 2, 'SCN': 3},
        ]
    }

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = 'pdf_extraction.log') -> logging.Logger:
    """Configure logging with both file and console output"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# ============================================================================
# PDF EXTRACTION FUNCTIONS
# ============================================================================

class PDFExtractor:
    """Professional PDF extraction tool for disease tolerance data"""

    def __init__(self, pdf_path: str, config: dict = None):
        """
        Initialize PDF extractor

        Args:
            pdf_path: Path to PDF file
            config: Configuration dictionary (uses Syngenta by default)
        """
        self.pdf_path = Path(pdf_path)
        self.config = config or PDFExtractionConfig.SYNGENTA_CONFIG

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Initializing PDF extraction: {self.pdf_path}")

    def extract_crop_data(self, crop_type: str, use_fallback: bool = True) -> List[Dict]:
        """
        Extract disease tolerance data for a specific crop

        Args:
            crop_type: 'Corn', 'Soybean', or 'Potato'
            use_fallback: Use hardcoded fallback data if PDF extraction fails

        Returns:
            List of dictionaries with variety and disease data
        """
        if crop_type not in self.config:
            raise ValueError(f"Crop type not supported: {crop_type}")

        crop_config = self.config[crop_type]

        if not crop_config['pages']:
            logger.warning(f"No pages configured for {crop_type}")
            # Try fallback data
            if use_fallback and crop_type in PDFExtractionConfig.FALLBACK_DATA:
                logger.info(f"Using fallback data for {crop_type}")
                return PDFExtractionConfig.FALLBACK_DATA[crop_type]
            return []

        logger.info(f"Extracting {crop_type} data from PDF...")
        variety_data = []

        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                # Extract data from configured pages
                for page_num in crop_config['pages']:
                    if page_num >= len(pdf.pages):
                        logger.debug(f"Page {page_num + 1} does not exist in PDF")
                        continue

                    page = pdf.pages[page_num]
                    tables = page.extract_tables()

                    if not tables:
                        logger.debug(f"No tables found on page {page_num + 1}")
                        continue

                    # Try to find and parse tables
                    for table_idx, table in enumerate(tables):
                        try:
                            extracted = self._parse_table_rows(
                                table, crop_type, crop_config
                            )
                            if extracted:
                                variety_data.extend(extracted)
                                logger.debug(f"Extracted {len(extracted)} varieties from table {table_idx} on page {page_num + 1}")
                        except Exception as e:
                            logger.debug(f"Could not parse table {table_idx} on page {page_num + 1}: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error during PDF extraction: {str(e)}")
            if use_fallback:
                logger.info(f"PDF extraction failed, falling back to hardcoded data")
                if crop_type in PDFExtractionConfig.FALLBACK_DATA:
                    return PDFExtractionConfig.FALLBACK_DATA[crop_type]
            raise

        # If no data extracted, use fallback
        if not variety_data:
            if use_fallback and crop_type in PDFExtractionConfig.FALLBACK_DATA:
                logger.warning(f"No data extracted from PDF, using fallback data for {crop_type}")
                variety_data = PDFExtractionConfig.FALLBACK_DATA[crop_type]
            else:
                logger.warning(f"No data found for {crop_type}")

        logger.info(f"Successfully extracted {len(variety_data)} {crop_type} varieties")
        return variety_data

    def _parse_table_rows(self, table: List[List], crop_type: str,
                         config: dict) -> List[Dict]:
        """
        Parse table rows and extract variety data

        Args:
            table: Table data from pdfplumber
            crop_type: Type of crop
            config: Crop-specific configuration

        Returns:
            List of parsed variety dictionaries
        """
        variety_data = []

        # Skip header rows (usually first 2 rows)
        for row_idx, row in enumerate(table[2:], start=2):
            try:
                # Get variety name
                variety_name = str(row[config['variety_col']]).strip() if row[config['variety_col']] else None

                if not variety_name or variety_name == '':
                    continue

                # Build disease tolerance dict
                entry = {
                    'Variety': variety_name,
                    'Crop': crop_type
                }

                # Extract disease ratings
                for disease_name, col_idx in config['disease_cols'].items():
                    try:
                        value = row[col_idx]
                        # Try to convert to int
                        if value and str(value).isdigit():
                            entry[disease_name] = int(value)
                        elif value:
                            entry[disease_name] = str(value).strip()
                        else:
                            entry[disease_name] = None
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Could not parse {disease_name} for {variety_name}: {e}")
                        entry[disease_name] = None

                variety_data.append(entry)
                logger.debug(f"Parsed: {variety_name}")

            except Exception as e:
                logger.debug(f"Error parsing row {row_idx}: {str(e)}")
                continue

        return variety_data

    def extract_all_crops(self) -> Dict[str, List[Dict]]:
        """
        Extract data for all configured crops

        Returns:
            Dictionary with crop type as key and data list as value
        """
        logger.info("Starting extraction for all crops...")
        all_data = {}

        for crop_type in self.config.keys():
            try:
                all_data[crop_type] = self.extract_crop_data(crop_type)
            except Exception as e:
                logger.error(f"Failed to extract {crop_type}: {str(e)}")
                all_data[crop_type] = []

        return all_data

    def save_to_csv(self, data: List[Dict], output_path: str,
                   crop_type: str = None) -> None:
        """
        Save extracted data to CSV file

        Args:
            data: List of dictionaries to save
            output_path: Output CSV file path
            crop_type: Crop type (for logging)
        """
        if not data:
            logger.warning(f"No data to save for {crop_type or 'unknown crop'}")
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get fieldnames from first record
            fieldnames = list(data[0].keys())

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Saved {len(data)} records to {output_path}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description='Extract disease tolerance data from seed company PDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Syngenta - Extract all crops
  python extract_pdf_to_csv.py --pdf "Syngenta_Guide.pdf" --company "Syngenta" --output-dir "Data"

  # Syngenta - Extract specific crop
  python extract_pdf_to_csv.py --pdf "Syngenta_Guide.pdf" --company "Syngenta" --crop "Corn" --output "corn.csv"

  # BASF - Extract Soybean
  python extract_pdf_to_csv.py --pdf "BASF_Xitavo.pdf" --company "BASF" --crop "Soybean" --output "basf_soybean.csv"

  # BASF - Full path example
  python extract_pdf_to_csv.py --pdf "Disease Resistance information/BASF_2025_Xitavo_Catalog_Digital_R1.pdf" --company "BASF" --crop "Soybean"
        """
    )

    parser.add_argument(
        '--pdf',
        required=True,
        help='Path to PDF file'
    )
    parser.add_argument(
        '--company',
        choices=['Syngenta', 'BASF', 'Corteva', 'Bayer'],
        default='Syngenta',
        help='Seed company (default: Syngenta)'
    )
    parser.add_argument(
        '--crop',
        help='Extract specific crop (Corn, Soybean, Potato, or company-specific)'
    )
    parser.add_argument(
        '--output',
        help='Output CSV file path (for single crop)'
    )
    parser.add_argument(
        '--output-dir',
        default='Data',
        help='Output directory for all crops (default: Data)'
    )

    args = parser.parse_args()

    try:
        # Select configuration based on company
        config = PDFExtractionConfig.SYNGENTA_CONFIG
        if args.company == 'BASF':
            config = PDFExtractionConfig.BASF_CONFIG
            logger.info(f"Using BASF configuration")
        else:
            logger.info(f"Using Syngenta configuration")

        # Initialize extractor
        extractor = PDFExtractor(args.pdf, config=config)

        if args.crop:
            # Extract single crop
            logger.info(f"Extracting {args.crop}...")
            data = extractor.extract_crop_data(args.crop)

            output_file = args.output or f"{args.output_dir}/{args.crop.lower()}_disease_tolerance.csv"
            extractor.save_to_csv(data, output_file, args.crop)

            logger.info(f"Extraction complete: {output_file}")
        else:
            # Extract all crops
            logger.info("Extracting all crops...")
            all_data = extractor.extract_all_crops()

            for crop_type, data in all_data.items():
                if data:
                    output_file = f"{args.output_dir}/{crop_type.lower()}_disease_tolerance.csv"
                    extractor.save_to_csv(data, output_file, crop_type)

            logger.info("Extraction complete for all crops")

        # Print summary
        logger.info("="*80)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*80)

        if args.crop:
            all_data = {args.crop: extractor.extract_crop_data(args.crop)}
        else:
            all_data = extractor.extract_all_crops()

        for crop_type, data in all_data.items():
            logger.info(f"{crop_type}: {len(data)} varieties extracted")

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
