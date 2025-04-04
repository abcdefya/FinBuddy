"""
Code Manipulation Utilities

This module provides utilities for code execution, file manipulation, and
visualization in Jupyter environments. It includes robust error handling
and follows best practices for file system operations.
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from typing_extensions import Annotated
from IPython import get_ipython
from IPython.display import display, Image
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for code operations
DEFAULT_CODE_DIR = Path("coding/")


@dataclass
class ExecutionResult:
    """Container for code execution results with status information."""
    success: bool
    output: str
    error_message: Optional[str] = None


class IPythonUtils:
    """
    Utilities for IPython/Jupyter notebook interaction.
    Provides methods for executing Python code and displaying media in notebooks.
    """

    @staticmethod
    def exec_python(
        cell: Annotated[str, "Valid Python code to execute."]
    ) -> ExecutionResult:
        """
        Execute Python code in the current IPython environment with comprehensive error handling.
        
        Args:
            cell: Python code to execute
            
        Returns:
            ExecutionResult with execution status and output
        """
        ipython = get_ipython()
        if not ipython:
            error_msg = "No active IPython environment found. This function requires a Jupyter notebook."
            logger.error(error_msg)
            return ExecutionResult(success=False, output="", error_message=error_msg)
            
        try:
            # Execute the code in the IPython environment
            result = ipython.run_cell(cell)
            
            # Collect output and errors
            output = str(result.result) if result.result else ""
            errors = []
            
            if result.error_before_exec is not None:
                errors.append(f"Syntax error: {result.error_before_exec}")
            if result.error_in_exec is not None:
                errors.append(f"Runtime error: {result.error_in_exec}")
                
            error_message = "\n".join(errors) if errors else None
            success = error_message is None
            
            return ExecutionResult(
                success=success,
                output=output,
                error_message=error_message
            )
            
        except Exception as e:
            error_msg = f"Unexpected error during code execution: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return ExecutionResult(success=False, output="", error_message=error_msg)

    @staticmethod
    def display_image(
        image_path: Annotated[str, "Path to image file to display."]
    ) -> ExecutionResult:
        """
        Display an image in the Jupyter notebook with error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ExecutionResult with display status and output
        """
        # Ensure path exists
        full_path = Path(image_path)
        if not full_path.exists():
            error_msg = f"Image not found: {image_path}"
            logger.error(error_msg)
            return ExecutionResult(success=False, output="", error_message=error_msg)
            
        # Check file type
        if full_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']:
            error_msg = f"Unsupported image format: {full_path.suffix}"
            logger.error(error_msg)
            return ExecutionResult(success=False, output="", error_message=error_msg)
            
        try:
            # Display image directly instead of executing code
            display(Image(filename=str(full_path)))
            return ExecutionResult(success=True, output="Image displayed successfully")
        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            logger.error(error_msg)
            return ExecutionResult(success=False, output="", error_message=error_msg)


class CodeFileManager:
    """
    Advanced utilities for file management and code manipulation.
    Provides methods for listing, reading, modifying, and creating code files.
    """

    def __init__(self, base_dir: Union[str, Path] = DEFAULT_CODE_DIR):
        """
        Initialize with base directory for operations.
        
        Args:
            base_dir: Base directory for code operations
        """
        self.base_dir = Path(base_dir)
        self._ensure_base_dir_exists()
        
    def _ensure_base_dir_exists(self) -> None:
        """Create base directory if it doesn't exist."""
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created base directory: {self.base_dir}")

    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """
        Resolve a file path relative to the base directory.
        
        Args:
            file_path: Relative or absolute path
            
        Returns:
            Resolved Path object
        """
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def list_dir(
        self, 
        directory: Annotated[str, "Directory to list files from."] = ""
    ) -> Dict[str, List[str]]:
        """
        List files and directories in the specified directory with structured output.
        
        Args:
            directory: Directory to list (relative to base_dir)
            
        Returns:
            Dictionary with 'files' and 'directories' keys
        """
        dir_path = self._resolve_path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return {"files": [], "directories": [], "error": f"Directory not found: {dir_path}"}
            
        if not dir_path.is_dir():
            logger.warning(f"Path is not a directory: {dir_path}")
            return {"files": [], "directories": [], "error": f"Not a directory: {dir_path}"}
            
        try:
            files = []
            directories = []
            
            for item in dir_path.iterdir():
                if item.is_file():
                    files.append(str(item.name))
                elif item.is_dir():
                    directories.append(str(item.name))
                    
            return {
                "files": sorted(files),
                "directories": sorted(directories)
            }
        except Exception as e:
            error_msg = f"Error listing directory {dir_path}: {str(e)}"
            logger.error(error_msg)
            return {"files": [], "directories": [], "error": error_msg}

    def read_file(
        self,
        file_path: Annotated[str, "Path to file to read."],
        line_numbers: Annotated[bool, "Include line numbers in output."] = True,
        start_line: Annotated[Optional[int], "First line to read (1-indexed)."] = None,
        end_line: Annotated[Optional[int], "Last line to read (1-indexed)."] = None
    ) -> Dict[str, Any]:
        """
        Read file contents with line numbers and range support.
        
        Args:
            file_path: Path to the file
            line_numbers: Whether to include line numbers in output
            start_line: First line to read (1-indexed, optional)
            end_line: Last line to read (1-indexed, optional)
            
        Returns:
            Dictionary with file information and contents
        """
        path = self._resolve_path(file_path)
        
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
            
        if not path.is_file():
            error_msg = f"Path is not a file: {path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
            
        try:
            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
            # Handle line range if specified
            if start_line is not None:
                start_idx = max(0, start_line - 1)  # Convert to 0-indexed
            else:
                start_idx = 0
                
            if end_line is not None:
                end_idx = min(len(lines), end_line)  # Convert to 0-indexed + 1
            else:
                end_idx = len(lines)
                
            # Slice lines and format with line numbers if requested
            selected_lines = lines[start_idx:end_idx]
            
            if line_numbers:
                formatted_lines = [f"{i+start_idx+1}: {line}" for i, line in enumerate(selected_lines)]
                content = "".join(formatted_lines)
            else:
                content = "".join(selected_lines)
                
            return {
                "success": True,
                "path": str(path),
                "total_lines": len(lines),
                "displayed_lines": len(selected_lines),
                "content": content
            }
            
        except Exception as e:
            error_msg = f"Error reading file {path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def modify_file(
        self,
        file_path: Annotated[str, "Path to file to modify."],
        start_line: Annotated[int, "First line to replace (1-indexed)."],
        end_line: Annotated[int, "Last line to replace (1-indexed)."],
        new_code: Annotated[str, "New code to insert."],
        create_backups: Annotated[bool, "Create backup before modifying."] = True
    ) -> Dict[str, Any]:
        """
        Modify a file by replacing lines with new code. Creates backups by default.
        
        Args:
            file_path: Path to the file
            start_line: First line to replace (1-indexed)
            end_line: Last line to replace (1-indexed)
            new_code: New code to insert
            create_backups: Whether to create a backup before modifying
            
        Returns:
            Dictionary with modification status and information
        """
        path = self._resolve_path(file_path)
        
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
            
        if not path.is_file():
            error_msg = f"Path is not a file: {path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
            
        # Validate line numbers
        if start_line < 1:
            return {"success": False, "error": f"Invalid start_line: {start_line} (must be ≥ 1)"}
            
        if end_line < start_line:
            return {"success": False, "error": f"end_line ({end_line}) must be ≥ start_line ({start_line})"}
            
        try:
            # Read the original content
            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
            # Validate line range
            if start_line > len(lines) + 1:
                return {"success": False, "error": f"start_line ({start_line}) exceeds file length ({len(lines)})"}
                
            # Create backup if requested
            if create_backups:
                backup_path = path.with_suffix(f"{path.suffix}.bak")
                with open(backup_path, "w", encoding="utf-8") as backup_file:
                    backup_file.writelines(lines)
                logger.info(f"Created backup at {backup_path}")
                
            # Adjust line indices (convert to 0-indexed)
            start_idx = start_line - 1
            end_idx = min(end_line, len(lines) + 1)
            
            # Ensure new_code ends with newline if not empty
            if new_code and not new_code.endswith("\n"):
                new_code += "\n"
                
            # Replace content
            modified_lines = lines[:start_idx] + [new_code] + lines[end_idx:]
            
            # Write back to file
            with open(path, "w", encoding="utf-8") as file:
                file.writelines(modified_lines)
                
            return {
                "success": True,
                "path": str(path),
                "lines_replaced": end_idx - start_idx,
                "backup_created": create_backups,
                "backup_path": str(backup_path) if create_backups else None
            }
            
        except Exception as e:
            error_msg = f"Error modifying file {path}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def create_file(
        self,
        file_path: Annotated[str, "Path to file to create."],
        content: Annotated[str, "Content to write to file."],
        overwrite: Annotated[bool, "Overwrite if file exists."] = False
    ) -> Dict[str, Any]:
        """
        Create a new file with the specified content.
        
        Args:
            file_path: Path to the file
            content: Content to write to the file
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with creation status and information
        """
        path = self._resolve_path(file_path)
        
        # Check if file already exists
        if path.exists() and not overwrite:
            error_msg = f"File already exists: {path}. Set overwrite=True to replace."
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
            
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
                
            return {
                "success": True,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "action": "overwritten" if (path.exists() and overwrite) else "created"
            }
            
        except Exception as e:
            error_msg = f"Error creating file {path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


# Backward compatibility functions with improved implementation
class CodingUtils:
    """
    Legacy interface for code file operations.
    Wraps CodeFileManager for backward compatibility.
    """

    _manager = CodeFileManager(DEFAULT_CODE_DIR)

    @classmethod
    def list_dir(cls, directory: Annotated[str, "Directory to check."]) -> str:
        """
        List files in chosen directory.
        
        Args:
            directory: Directory to list
            
        Returns:
            String representation of files and directories
        """
        result = cls._manager.list_dir(directory)
        if "error" in result:
            return f"Error: {result['error']}"
            
        files = result.get("files", [])
        dirs = result.get("directories", [])
        
        output = []
        if dirs:
            output.append("Directories:")
            output.extend([f"  - {d}" for d in dirs])
            
        if files:
            output.append("Files:")
            output.extend([f"  - {f}" for f in files])
            
        if not output:
            return "Directory is empty"
            
        return "\n".join(output)

    @classmethod
    def see_file(cls, filename: Annotated[str, "Name and path of file to check."]) -> str:
        """
        Check the contents of a chosen file.
        
        Args:
            filename: Path to file to read
            
        Returns:
            File contents with line numbers
        """
        result = cls._manager.read_file(filename)
        if not result.get("success"):
            return f"Error: {result.get('error')}"
            
        return result.get("content", "")

    @classmethod
    def modify_code(
        cls,
        filename: Annotated[str, "Name and path of file to change."],
        start_line: Annotated[int, "Start line number to replace with new code."],
        end_line: Annotated[int, "End line number to replace with new code."],
        new_code: Annotated[str, "New piece of code to replace old code with."],
    ) -> str:
        """
        Replace old piece of code with new one. Proper indentation is important.
        
        Args:
            filename: Path to file
            start_line: First line to replace
            end_line: Last line to replace
            new_code: New code to insert
            
        Returns:
            Status message
        """
        result = cls._manager.modify_file(filename, start_line, end_line, new_code)
        if not result.get("success"):
            return f"Error: {result.get('error')}"
            
        lines_replaced = result.get("lines_replaced", 0)
        return f"Code modified successfully. Replaced {lines_replaced} line(s)."

    @classmethod
    def create_file_with_code(
        cls,
        filename: Annotated[str, "Name and path of file to create."],
        code: Annotated[str, "Code to write in the file."],
    ) -> str:
        """
        Create a new file with provided code.
        
        Args:
            filename: Path to file
            code: Content to write
            
        Returns:
            Status message
        """
        result = cls._manager.create_file(filename, code)
        if not result.get("success"):
            return f"Error: {result.get('error')}"
            
        return f"File created successfully: {result.get('path')}"


