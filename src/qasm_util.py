import re

def expand_qasm(text) :

    # Define a regular expression pattern to match the desired substrings
    pattern = r'gate (\w+) ([^{}]+) \{([^}]+)\}'
    matching = True
    while matching:
        match = re.search(pattern, text)

        if match:
            # Extract the entire matched string
            matched_string = match.group(0)
            # Remove the matched string from the original text
            text = text.replace(matched_string, "", 1)  # Remove only the first occurrence

            # Extract the captured groups
            gate_name = match.group(1)
            params = match.group(2).split(",")
            gate_body = match.group(3)

            application_pattern = f'{gate_name} ([^{{}};]+);'
            application_matching = True

            while application_matching:
                application_match = re.search(application_pattern, text)

                if application_match:
                    application_string = application_match.group(0)
                    arguments = application_match.group(1).split(",")
                    application_text = gate_body
                    
                    for i, param in enumerate(params):
                        application_text = application_text.replace(param.strip(), arguments[i])
                    
                    text = text.replace(application_string, application_text)
                else:
                    application_matching = False
        else:
            matching = False

    return re.sub(r'\s+', ' ', text)





