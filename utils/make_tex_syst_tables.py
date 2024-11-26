# Sample data: Lists for each column
source_of_systematic = ['Syst1', 'Syst2', 'Syst3']
nominal = [1.23, 4.56, 7.89]
prior = [0.12, 0.34, 0.56]
fit_value = [1.11, 4.44, 7.77]

# Open a file to write the LaTeX code
latex_code = """
\\begin{table}[h!]
\\centering
\\begin{tabular}{c|c|c|c}
Source of systematic & Nominal & Prior & Fit value \\\\
\\hline
\\hline
"""

# Add rows from the lists
for i in range(len(source_of_systematic)):
    latex_code += f"{source_of_systematic[i]} & {nominal[i]} & {prior[i]} & {fit_value[i]} \\\\ \n"

# Closing the table
latex_code += """
\\hline
\\end{tabular}
\\caption{Systematic Table}
\\label{tab:systematic}
\\end{table}
"""

# Print the LaTeX code (you can also save this to a .tex file)
print(latex_code)
