from typing import TypedDict
from pydantic import BaseModel


class Person(TypedDict):
    name: str
    age: int

person = Person(name="John", age=30)
print(person)

class PersonModel(BaseModel):
  name: str
  age: int

person_model = PersonModel(name="John", age=30)
print(person_model)
