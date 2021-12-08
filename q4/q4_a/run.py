# %%
from typing import Union

def bro(the_string: str, the_union: Union[str, int]):
    print(f"the string: {the_string}")
    print(f"the union: {the_union}")

bro("hello", 123)
bro("hello", "world")
bro("hello", 123.456)

# %%
