import os

filepath = r"c:\Users\Deng Junxian\Desktop\Civitas_new\core\inference\api_backend.py"
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the eager client load 
text = text.replace('        client = self._get_client()\n        \n        messages = []', '        messages = []', 1)
if 'messages = []' not in text:
    text = text.replace('        client = self._get_client()\r\n        \r\n        messages = []', '        messages = []', 1)

# Replace the exception catch
text = text.replace(
'''        except Exception as e:
            return f"[API Error] {e}"''',
'''        except Exception as e:
            fallback = kwargs.get("fallback_response")
            if fallback is not None:
                return str(fallback)
            return f"[API Error] {e}"''', 1)

if 'fallback_response' not in text:
    # Try CRLF just in case
    old_except = '        except Exception as e:\r\n            return f"[API Error] {e}"'
    new_except = '        except Exception as e:\r\n            fallback = kwargs.get("fallback_response")\r\n            if fallback is not None:\r\n                return str(fallback)\r\n            return f"[API Error] {e}"'
    text = text.replace(old_except, new_except, 1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed api_backend.py")
