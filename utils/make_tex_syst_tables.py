# Sample data: Lists for each column
source_of_systematic = ['Normalization < 1 GeV', 'Normalization > 1 GeV', 'Zenith up-going', 'Zenith down-going', r'$\nu / \overbar{\nu}$', r'$e/\mu$','Spectral index']
nominal = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
prior = [0.12, 0.34, 0.56]
fit_value = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]

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
