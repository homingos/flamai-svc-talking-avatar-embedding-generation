import requests
import pandas as pd
import csv
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import io
import logging
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CSVReadResult:
    """Data model for CSV reading result"""
    success: bool
    texts: List[str]
    total_rows: int
    columns_processed: List[str]
    processing_time: float
    source_url: str
    message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class CSVTextReader:
    """
    CSV Text Reader for extracting text content from CSV files via URL
    Integrated with the embedding service architecture
    """
    
    def __init__(self, timeout: int = 30, max_file_size: int = 50 * 1024 * 1024):
        """
        Initialize CSV Text Reader
        
        Args:
            timeout: Request timeout in seconds
            max_file_size: Maximum file size in bytes (default 50MB)
        """
        self.timeout = timeout
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)
        
    def read_csv_from_url(
        self, 
        url: str, 
        text_columns: Optional[List[str]] = None,
        combine_columns: bool = True,
        separator: str = " ",
        encoding: str = 'utf-8',
        skip_empty: bool = True
    ) -> CSVReadResult:
        """
        Read CSV file from URL and extract text content
        
        Args:
            url: URL to the CSV file
            text_columns: Specific columns to extract text from (None = all columns)
            combine_columns: Whether to combine multiple columns into single text
            separator: Separator when combining columns
            encoding: File encoding
            skip_empty: Whether to skip empty rows/cells
            
        Returns:
            CSVReadResult: Result containing extracted texts and metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return CSVReadResult(
                    success=False,
                    texts=[],
                    total_rows=0,
                    columns_processed=[],
                    processing_time=0.0,
                    source_url=url,
                    message="Invalid URL provided",
                    error_details={"error_type": "ValidationError"}
                )
            
            # Download CSV file
            csv_content = self._download_csv(url, encoding)
            if csv_content is None:
                return CSVReadResult(
                    success=False,
                    texts=[],
                    total_rows=0,
                    columns_processed=[],
                    processing_time=0.0,
                    source_url=url,
                    message="Failed to download CSV file",
                    error_details={"error_type": "DownloadError"}
                )
            
            # Parse CSV and extract texts
            texts, columns_processed, total_rows = self._extract_texts(
                csv_content, text_columns, combine_columns, separator, skip_empty
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CSVReadResult(
                success=True,
                texts=texts,
                total_rows=total_rows,
                columns_processed=columns_processed,
                processing_time=processing_time,
                source_url=url,
                message=f"Successfully extracted {len(texts)} texts from CSV"
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error reading CSV from {url}: {str(e)}")
            
            return CSVReadResult(
                success=False,
                texts=[],
                total_rows=0,
                columns_processed=[],
                processing_time=processing_time,
                source_url=url,
                message=f"Error processing CSV: {str(e)}",
                error_details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _download_csv(self, url: str, encoding: str) -> Optional[str]:
        """Download CSV file from URL"""
        try:
            headers = {
                'User-Agent': 'Embedding-Service-CSV-Reader/1.0',
                'Accept': 'text/csv, application/csv, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            response = requests.get(url, timeout=self.timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise ValueError(f"File too large: {content_length} bytes > {self.max_file_size} bytes")
            
            # Download with size limit
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_file_size:
                    raise ValueError(f"File too large: > {self.max_file_size} bytes")
            
            # Try to decode with specified encoding, fallback to common encodings
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                for fallback_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return content.decode(fallback_encoding)
                    except UnicodeDecodeError:
                        continue
                # If all encodings fail, use utf-8 with error handling
                return content.decode('utf-8', errors='replace')
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error downloading {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {str(e)}")
            return None
    
    def _extract_texts(
        self, 
        csv_content: str, 
        text_columns: Optional[List[str]], 
        combine_columns: bool, 
        separator: str, 
        skip_empty: bool
    ) -> tuple[List[str], List[str], int]:
        """Extract text content from CSV"""
        try:
            # Use pandas for robust CSV parsing
            df = pd.read_csv(io.StringIO(csv_content))
            total_rows = len(df)
            
            # Determine columns to process
            if text_columns:
                # Use specified columns
                available_columns = [col for col in text_columns if col in df.columns]
                if not available_columns:
                    self.logger.warning(f"None of the specified columns {text_columns} found in CSV")
                    # Fall back to all columns if specified columns not found
                    columns_to_use = df.columns.tolist()
                else:
                    columns_to_use = available_columns
            else:
                # Use all columns
                columns_to_use = df.columns.tolist()
            
            texts = []
            
            for _, row in df.iterrows():
                if combine_columns:
                    # Combine multiple columns into single text
                    row_texts = []
                    for col in columns_to_use:
                        cell_value = row[col]
                        if pd.notna(cell_value) and str(cell_value).strip():
                            row_texts.append(str(cell_value).strip())
                    
                    if row_texts or not skip_empty:
                        combined_text = separator.join(row_texts)
                        if combined_text.strip() or not skip_empty:
                            texts.append(combined_text)
                else:
                    # Keep each column as separate text
                    for col in columns_to_use:
                        cell_value = row[col]
                        if pd.notna(cell_value):
                            cell_text = str(cell_value).strip()
                            if cell_text or not skip_empty:
                                texts.append(cell_text)
            
            return texts, columns_to_use, total_rows
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV content with pandas: {str(e)}")
            # Fallback to basic CSV parsing
            return self._extract_texts_fallback(csv_content, skip_empty)
    
    def _extract_texts_fallback(self, csv_content: str, skip_empty: bool) -> tuple[List[str], List[str], int]:
        """Fallback CSV parsing method using basic csv module"""
        texts = []
        columns_processed = []
        total_rows = 0
        
        try:
            csv_reader = csv.reader(io.StringIO(csv_content))
            
            # Read header
            header = next(csv_reader, None)
            if header:
                columns_processed = header
            
            # Read data rows
            for row in csv_reader:
                total_rows += 1
                for cell in row:
                    if cell and cell.strip():
                        texts.append(cell.strip())
                    elif not skip_empty:
                        texts.append(cell if cell else "")
            
            return texts, columns_processed, total_rows
            
        except Exception as e:
            self.logger.error(f"Fallback CSV parsing also failed: {str(e)}")
            return [], [], 0


# Convenience functions for integration
def extract_texts_from_csv_url(
    url: str, 
    text_columns: Optional[List[str]] = None,
    combine_columns: bool = True,
    separator: str = " ",
    skip_empty: bool = True
) -> List[str]:
    """
    Simple function to extract texts from CSV URL
    Returns only the list of texts for direct use with embedding requests
    
    Args:
        url: URL to the CSV file
        text_columns: Specific columns to extract (None = all columns)
        combine_columns: Whether to combine columns into single text
        separator: Separator when combining columns
        skip_empty: Whether to skip empty cells
        
    Returns:
        List[str]: List of extracted texts
        
    Raises:
        Exception: If CSV processing fails
    """
    reader = CSVTextReader()
    result = reader.read_csv_from_url(
        url=url,
        text_columns=text_columns,
        combine_columns=combine_columns,
        separator=separator,
        skip_empty=skip_empty
    )
    
    if result.success:
        return result.texts
    else:
        raise Exception(f"Failed to process CSV: {result.message}")