from typing import Optional

# Simple data structures using dicts

def create_lesson(lesson_number: int, title: str, lesson_link: Optional[str] = None) -> dict:
    """Create a lesson dictionary"""
    return {
        "lesson_number": lesson_number,
        "title": title,
        "lesson_link": lesson_link
    }

def create_course(title: str, course_link: Optional[str] = None,
                 instructor: Optional[str] = None, lessons: Optional[list] = None) -> dict:
    """Create a course dictionary"""
    return {
        "title": title,
        "course_link": course_link,
        "instructor": instructor,
        "lessons": lessons if lessons is not None else []
    }

def create_course_chunk(content: str, course_title: str,
                       lesson_number: Optional[int] = None, chunk_index: int = 0) -> dict:
    """Create a course chunk dictionary"""
    return {
        "content": content,
        "course_title": course_title,
        "lesson_number": lesson_number,
        "chunk_index": chunk_index
    }

def create_message(role: str, content: str) -> dict:
    """Create a message dictionary"""
    return {
        "role": role,
        "content": content
    }
