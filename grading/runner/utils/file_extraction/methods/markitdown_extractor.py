"""
MarkItDown-based file extraction implementation.
"""

from io import BytesIO
from pathlib import Path

from loguru import logger
from markitdown import MarkItDown

from ..base import BaseFileExtractor
from ..types import ExtractedContent


class MarkItDownExtractor(BaseFileExtractor):
    """
    File extractor using MarkItDown for document parsing.

    Supports: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".csv",
    }

    def __init__(self):
        self._converter = MarkItDown()

    async def extract_from_file(
        self,
        file_path: Path,
        *,
        include_images: bool = True,
        sub_artifact_index: int | None = None,
    ) -> ExtractedContent:
        """
        Extract content from a document using MarkItDown.

        Notes:
        - MarkItDown currently provides text content only in this integration.
        - `sub_artifact_index` is ignored by MarkItDown and included in metadata.
        """
        _ = include_images  # MarkItDown integration currently returns text-only output.

        try:
            with open(file_path, "rb") as f:
                document = self._converter.convert(BytesIO(f.read()))

            text = getattr(document, "text_content", None)
            if not text:
                # Defensive fallback for future/alternate object shapes.
                text = str(document) if document is not None else ""

            return ExtractedContent(
                text=text,
                images=[],
                extraction_method="markitdown",
                metadata={
                    "file_type": file_path.suffix,
                    "sub_artifact_index": sub_artifact_index,
                    "sub_artifact_index_supported": False,
                },
                sub_artifacts=[],
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract content from {file_path} using MarkItDown: {type(e).__name__}: {e}"
            )
            raise

    def supports_file_type(self, file_extension: str) -> bool:
        return file_extension.lower() in self.SUPPORTED_EXTENSIONS

    @property
    def name(self) -> str:
        return "markitdown"
