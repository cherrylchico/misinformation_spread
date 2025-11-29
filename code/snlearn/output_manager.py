"""
Output Manager for organizing simulation outputs with timestamps
"""

import os
from datetime import datetime
from pathlib import Path


class OutputManager:
    """Manages output directory creation with timestamps"""
    
    _current_output_dir = None
    
    @classmethod
    def create_output_dir(cls, base_dir='output'):
        """
        Create a timestamped output directory
        
        Args:
            base_dir: Base directory name (default: 'output')
            
        Returns:
            Path to the created output directory
        """
        # Get project root (assuming this file is in code/snlearn/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        
        # Create timestamp in format YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create full path: project_root/output/YYYYMMDD_HHMMSS_output
        output_base = project_root / base_dir
        output_dir = output_base / f"{timestamp}_output"
        
        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store for later use
        cls._current_output_dir = output_dir
        
        return str(output_dir)
    
    @classmethod
    def get_output_dir(cls):
        """
        Get the current output directory
        
        Returns:
            Path to current output directory, or None if not created yet
        """
        return cls._current_output_dir
    
    @classmethod
    def get_output_path(cls, filename):
        """
        Get full path for an output file
        
        Args:
            filename: Name of the output file
            
        Returns:
            Full path to the output file
        """
        if cls._current_output_dir is None:
            cls.create_output_dir()
        
        return str(Path(cls._current_output_dir) / filename)
