import re

with open('README.md', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

new_lines = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('![') and '](' in line:
        images = []
        while i < len(lines) and lines[i].strip().startswith('![') and '](' in lines[i].strip():
            img_line = lines[i].strip()
            match = re.match(r'\!\[([^\]]*)\]\(([^)]+)\)', img_line)
            if match:
                alt_text, src = match.groups()
                images.append((alt_text, src))
            i += 1
            
        caption = []
        if i < len(lines) and lines[i].strip().startswith('*') and not lines[i].strip().startswith('**'):
            cap_line = lines[i].strip()
            if cap_line.endswith('*'):
                caption.append(cap_line[1:-1])
                i += 1
            else:
                caption.append(cap_line[1:])
                i += 1
                while i < len(lines):
                    cont_line = lines[i].strip()
                    if cont_line.endswith('*'):
                        caption.append(cont_line[:-1])
                        i += 1
                        break
                    else:
                        caption.append(cont_line)
                        i += 1

        new_lines.append('<p align="center">')
        for alt, src in images:
            new_lines.append(f'  <img src="{src}" alt="{alt}">')
        if caption:
            cap_text = " ".join(caption)
            new_lines.append('  <br>')
            new_lines.append(f'  <em>{cap_text}</em>')
        new_lines.append('</p>')
    else:
        new_lines.append(lines[i])
        i += 1

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))
