from typing import TypedDict, Union, Optional
from pydantic import BaseModel

class Movie(TypedDict):
    name: str
    year: int
    rating: Union[int, float]
    director: Optional[str] = None
    

class PyMovie(BaseModel):
    name: str
    year: int
    rating: Union[int, float]
    director: Optional[str] = None
    
def square(x: Union[int, float]) -> Union[int, float]:
    return x * x

def some_func(name: Optional[str] = None):
    if name:
        return f"Hello, {name}"
    return "Hello, World!"

def main():
    movie = Movie(
        name="The Matrix",
        year=1999,
        rating="5.0",
    )
    print(movie)

    py_movie = PyMovie(
        name="The Matrix",
        year=1999,
        rating="5.0",
    )
    print(py_movie)

    print(some_func())

if __name__ == "__main__":
    main()
