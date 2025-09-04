"""Excel file processing utilities for RAG system."""

import json
from typing import List, Dict, Any, Tuple, Optional

from loguru import logger


async def process_excel_file(file_content: bytes, filename: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Process Excel file and extract text content for RAG ingestion.
    This version uses a simple approach since we know the structure.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Tuple of (text_chunks, metadata)
    """
    try:
        logger.info(f"Processing Excel file: {filename}")
        
        # For now, we'll create a placeholder and instructions for manual processing
        # In production, you would use openpyxl or xlrd here
        
        text_chunks = [
            f"Обнаружен Excel файл: {filename}",
            f"Размер файла: {len(file_content)} байт",
            "Для полной обработки Excel файлов установите библиотеки openpyxl или xlrd",
            "Или используйте специальный endpoint /ingest/egov-data для предварительно обработанных данных"
        ]
        
        metadata = {
            "source_type": "excel",
            "filename": filename,
            "file_size": len(file_content),
            "total_chunks": len(text_chunks),
            "note": "Excel файл обнаружен. Используйте /ingest/egov-data для структурированной загрузки"
        }
        
        return text_chunks, metadata
            
    except Exception as e:
        logger.error(f"Failed to process Excel file {filename}: {e}")
        raise ValueError(f"Could not process Excel file: {str(e)}")


def create_structured_content_from_egov_data(data: List[Dict[str, Any]]) -> List[str]:
    """
    Create structured text content from eGov services Excel data for better RAG performance.
    
    This function is specifically designed for eGov services data with columns:
    - id: Service ID
    - name: Service name
    - eGov_link: Link to Russian version
    - chunks: Detailed service description
    - eGov_kaz_link: Link to Kazakh version
    
    Args:
        data: List of row dictionaries from Excel
        
    Returns:
        List of formatted text chunks
    """
    text_chunks = []
    
    logger.info(f"Processing {len(data)} eGov services records")
    
    for i, row in enumerate(data, 1):
        try:
            # Extract key fields
            service_id = row.get('id', '')
            service_name = row.get('name', '').strip()
            chunks_content = row.get('chunks', '').strip()
            egov_link = row.get('eGov_link', '').strip()
            egov_kaz_link = row.get('eGov_kaz_link', '').strip()
            
            if not chunks_content:
                logger.warning(f"Пропускаем запись {i}: отсутствует основное содержимое")
                continue
            
            # Create structured content
            content_parts = []
            
            # Add service header
            if service_name:
                content_parts.append(f"=== ГОСУДАРСТВЕННАЯ УСЛУГА ===")
                content_parts.append(f"Название: {service_name}")
            
            if service_id:
                content_parts.append(f"Идентификатор услуги: {service_id}")
            
            # Add main content with better structure
            if chunks_content:
                content_parts.append("=== ПОДРОБНОЕ ОПИСАНИЕ ===")
                
                # Clean up the content - replace \r\n with proper line breaks
                cleaned_content = chunks_content.replace('\\r\\n', '\n').replace('\r\n', '\n')
                content_parts.append(cleaned_content)
            
            # Add reference links
            links_section = []
            if egov_link:
                links_section.append(f"• Официальная ссылка (русский): {egov_link}")
            
            if egov_kaz_link:
                links_section.append(f"• Официальная ссылка (казахский): {egov_kaz_link}")
            
            if links_section:
                content_parts.append("=== ОФИЦИАЛЬНЫЕ ССЫЛКИ ===")
                content_parts.extend(links_section)
            
            # Combine into single chunk
            full_content = "\n\n".join(content_parts)
            
            # Handle very large content by splitting intelligently
            if len(full_content) > 12000:  # Large content threshold
                logger.info(f"Разделяем большой документ для услуги {service_id}: {service_name}")
                
                # Create a header chunk
                header_parts = []
                if service_name:
                    header_parts.append(f"=== ГОСУДАРСТВЕННАЯ УСЛУГА ===")
                    header_parts.append(f"Название: {service_name}")
                if service_id:
                    header_parts.append(f"Идентификатор: {service_id}")
                
                header_chunk = "\n\n".join(header_parts)
                
                # Split main content by sections
                sections = cleaned_content.split('\n\n')
                
                current_chunk = header_chunk
                chunk_counter = 1
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                        
                    # If adding this section would make chunk too large
                    if len(current_chunk) + len(section) + 4 > 10000:
                        # Add chunk number and save current chunk
                        final_chunk = f"{current_chunk}\n\n=== ЧАСТЬ {chunk_counter} ===\n\n{section}"
                        if len(final_chunk) > 12000:
                            # Even single section is too large, save what we have
                            text_chunks.append(current_chunk + f"\n\n[Продолжение в следующей части...]")
                            current_chunk = f"{header_chunk}\n\n=== ЧАСТЬ {chunk_counter + 1} ===\n\n{section[:8000]}..."
                            chunk_counter += 2
                        else:
                            text_chunks.append(final_chunk)
                            current_chunk = header_chunk
                            chunk_counter += 1
                    else:
                        if current_chunk == header_chunk:
                            current_chunk += f"\n\n=== ЧАСТЬ {chunk_counter} ===\n\n{section}"
                        else:
                            current_chunk += f"\n\n{section}"
                
                # Add remaining chunk
                if current_chunk.strip() != header_chunk.strip():
                    text_chunks.append(current_chunk)
                
                # Add links as separate chunk if exists
                if links_section:
                    links_chunk = f"{header_chunk}\n\n=== ОФИЦИАЛЬНЫЕ ССЫЛКИ ===\n\n" + "\n".join(links_section)
                    text_chunks.append(links_chunk)
                    
            else:
                # Content is manageable size, add as single chunk
                text_chunks.append(full_content)
                
        except Exception as e:
            logger.error(f"Ошибка при обработке записи {i}: {e}")
            continue
    
    logger.info(f"Создано {len(text_chunks)} структурированных текстовых блоков из данных eGov")
    return text_chunks


def format_egov_service_metadata(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create metadata for eGov service record.
    
    Args:
        row_data: Single row data from Excel
        
    Returns:
        Formatted metadata dictionary
    """
    metadata = {
        "content_type": "egov_service",
        "service_id": row_data.get('id'),
        "service_name": row_data.get('name', '').strip(),
        "egov_link_ru": row_data.get('eGov_link', '').strip(),
        "egov_link_kz": row_data.get('eGov_kaz_link', '').strip(),
        "language": "ru",  # Primary language
        "source": "egov_portal_kazakhstan"
    }
    
    # Clean empty values
    return {k: v for k, v in metadata.items() if v}


def validate_egov_data_structure(data: List[Dict[str, Any]]) -> bool:
    """
    Validate that the data has the expected eGov structure.
    
    Args:
        data: List of dictionaries to validate
        
    Returns:
        True if data structure is valid, False otherwise
    """
    if not data or not isinstance(data, list):
        logger.error("Data must be a non-empty list")
        return False
    
    # Check required fields in first record
    required_fields = ['id', 'name', 'chunks']
    optional_fields = ['eGov_link', 'eGov_kaz_link']
    
    first_record = data[0]
    if not isinstance(first_record, dict):
        logger.error("Records must be dictionaries")
        return False
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in first_record]
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return False
    
    # Validate field types and content
    records_with_content = 0
    for i, record in enumerate(data):
        if record.get('chunks', '').strip():
            records_with_content += 1
        
        # Basic validation
        if not isinstance(record.get('id'), (int, str)):
            logger.warning(f"Record {i + 1}: invalid ID type")
        
        if not record.get('name', '').strip():
            logger.warning(f"Record {i + 1}: empty name")
    
    if records_with_content == 0:
        logger.error("No records contain content in 'chunks' field")
        return False
    
    logger.info(f"Validated {len(data)} records, {records_with_content} with content")
    return True


def clean_egov_text_content(text: str) -> str:
    """
    Clean and normalize text content from eGov data.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text content
    """
    if not text:
        return ""
    
    # Replace various line break representations
    cleaned = text.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n')
    
    # Normalize multiple line breaks
    import re
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    # Remove excessive whitespace but preserve structure
    lines = cleaned.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Remove empty lines at the beginning and end
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)