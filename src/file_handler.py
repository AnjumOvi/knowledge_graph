import json
from typing import Tuple, Any

def parse_uploaded_file(file) -> Tuple[str, Any]:
    """Parse uploaded txt or json file and return content and type."""
    filename = file.name
    if filename.endswith('.txt'):
        content = file.read().decode('utf-8')
        return 'txt', content
    elif filename.endswith('.json'):
        content = json.load(file)
        return 'json', content
    else:
        raise ValueError('Unsupported file type. Only .txt and .json are supported.') 