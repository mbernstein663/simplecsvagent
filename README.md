# Simple Practice Agent

Agents tend to have 3 levels of complexity. They are effectively just advanced conditional for loops that incorporate LLMs to make dynamic decision-making and operate tools based on certain conditions.

- Level 1: Simple logic-based workflow with no LLM. No dynamic decision making. Outputs + tools are contingent on set conditions.
- Level 2:  Simple workflow with LLM inference to make dynamic decisions, calls tools to produce output.
- Level 3: Autonomous loop powered by LLM which can decide what to do next and when it's done. These agents can chain actions.

Here we have a very simple Level 2 agent that reads a .csv and dynamically cleans it using one of a few simple python operations including:

- standardizing column names
- dropping duplicate rows
- dropping unnamed columns
- filling numeric nulls with median
- fill text nulls with unknown
- drop columns with othe same entry in each cell

Rather than being contingent on conditionals, the LLM makes inferences by reading the data and dynamically decides on operations for what seems contextually reasonable. (i.e. it won't drop every unnamed column if it seems to have important data)
By defining python operations, we are limiting the LLM's range of options for affecting the file.

To run:

```
venv\Scripts\activate
pip install requirement.txt
python llm_csv_agent.py
```