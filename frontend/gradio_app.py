# frontend/gradio_app.py
# Optimized, Streaming & Real-Time Gradio UI
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import httpx
import json
import time
from config.settings import settings

BASE_URL = f"http://localhost:{settings.langserve_port}"
# Expanded timeouts to handle first-load cold starts on CPU.
TIMEOUT  = httpx.Timeout(180.0, connect=10.0, read=180.0)

def stream_api(path: str, payload: dict):
    """Generator that yields chunks with improved error resilience for network drops and timeouts."""
    try:
        with httpx.stream("POST", f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT) as response:
            if response.status_code != 200:
                yield f"❌ Server Error {response.status_code}: {response.text}"
                return

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                
                try:
                    # LangServe can emit raw JSON strings or complex objects
                    data = json.loads(data_str)
                    
                    # 1. Handle direct content strings
                    if isinstance(data, str):
                        yield data
                    # 2. Handle LangServe 'ops' format (incremental updates)
                    elif isinstance(data, dict):
                        # Extract from 'content', 'output', or 'ops'
                        if "ops" in data:
                            for op in data["ops"]:
                                if op.get("op") == "add" and isinstance(op.get("value"), str):
                                    yield op["value"]
                        elif "content" in data:
                            yield data["content"]
                        elif "output" in data:
                            if isinstance(data["output"], str):
                                yield data["output"]
                            elif isinstance(data["output"], dict) and "content" in data["output"]:
                                yield data["output"]["content"]
                        else:
                            # Try to yield the whole dict as string if we can't find content
                            pass 
                except json.JSONDecodeError:
                    # Fallback for raw non-JSON text in the 'data:' field
                    yield data_str.strip('"').replace("\\n", "\n")
                    
    except (httpx.RemoteProtocolError, httpx.ReadError):
        yield "\n\n⚠️ *Connection lag detected. Showing partial response.*"
    except httpx.ReadTimeout:
        yield "\n\n❌ *Model timeout. Llama 3.2 might be taking too long to start or respond.*"
    except Exception as e:
        yield f"\n\n❌ *Stream Error: {str(e)}*"

def chat_fn(message, history, mode):
    if not message.strip(): return
    
    partial_msg = ""
    
    if mode == "RAG (Contract Q&A)":
        # 1. Retrieve (Sync, but fast)
        yield "🔍 *Searching document context...*"
        try:
            r = httpx.post(f"{BASE_URL}/retriever/invoke", json={"input": message}, timeout=TIMEOUT)
            r.raise_for_status()
            docs = r.json().get("output", [])
        except Exception as e:
            yield f"❌ Retrieval Error: {e}"
            return
            
        if not docs:
            ctx = "No relevant context found."
        else:
            # Formatter for the backend prompt
            ctx = []
            for d in docs:
                m = d.get('metadata', {})
                src = m.get('source', 'Unknown')
                content = d.get('page_content', '')
                ctx.append({"metadata": m, "page_content": content})
        
        # 2. Stream Generation
        yield "✍️ *Thinking...*"
        partial_msg = ""
        # The generator/stream endpoint expects {"input": {"input": message, "context": docs}}
        for chunk in stream_api("/generator/stream", {"input": {"input": message, "context": ctx}}):
            if chunk:
                partial_msg += str(chunk)
                yield partial_msg
            
        # 3. Append sources at the end
        if docs:
            src_list = sorted(list(set([d.get('metadata', {}).get('source') for d in docs if d.get('metadata', {}).get('source')])))
            if src_list:
                sources = "\n\n---\n**Sources:**\n" + "\n".join([f"- {s}" for s in src_list])
                partial_msg += sources
                yield partial_msg
    else:
        # Basic Chat
        yield "✍️ *Thinking...*"
        for chunk in stream_api("/basic_chat/stream", {"input": message}):
            if chunk:
                partial_msg += str(chunk)
                yield partial_msg

def handle_upload(file):
    if not file: return "⚠️ No file selected."
    try:
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f)}
            r = httpx.post(f"{BASE_URL}/ingest", files=files, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "success":
                return f"✅ **{data['file']}** ingested! ({data['chunks']} chunks found)."
            else:
                return f"❌ Ingestion issue: {data.get('status')} - {data.get('detail','')}"
    except Exception as e:
        return f"❌ Upload failed: {e}"

with gr.Blocks(title="Contract Assistant") as demo:
    gr.Markdown("Smart Contract Assistant")
    
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            mode_radio = gr.Radio(["RAG (Contract Q&A)", "Basic Chat"], value="RAG (Contract Q&A)", label="Mode")
            
            def chat_wrapper(msg, hist):
                for update in chat_fn(msg, hist, mode_radio.value):
                    yield update
            
            gr.ChatInterface(
                chat_wrapper,
                examples=["Summarize this contract", "What is the content of document?"],
                textbox=gr.Textbox(placeholder="Ask a question..."),
            )
            
        with gr.Tab("📁 Upload"):
            file_input = gr.File(label="Upload Contract (PDF/DOCX)")
            upload_btn = gr.Button("🚀 Ingest Document", variant="primary")
            upload_out = gr.Markdown("*Status will appear here...*")
            upload_btn.click(handle_upload, inputs=file_input, outputs=upload_out)

        with gr.Tab("🔧 Health & Eval"):
            gr.Markdown("### 🔍 System Diagnostics")
            with gr.Row():
                health_btn = gr.Button("Check Server Status")
                eval_btn = gr.Button("🚀 Run Synthetic Eval", variant="primary")
            
            health_out = gr.JSON(label="Server Health Status")
            eval_out = gr.HTML(label="Evaluation Metrics")
            
            def check_health():
                try:
                    r = httpx.get(f"{BASE_URL}/health", timeout=10)
                    return r.json()
                except Exception as e:
                    return {"status": "offline", "error": str(e)}
            
            def run_eval_stream():
                full_results = ""
                yield "🚀 *Starting Evaluation (LLM-as-a-judge)...*"
                for chunk in stream_api("/evaluate", {}):
                    if chunk:
                        full_results += str(chunk)
                        yield f"<div style='background: #1e1e1e; padding: 10px; border-radius: 5px; color: lightgreen; font-family: monospace;'>{full_results}</div>"
            
            health_btn.click(check_health, outputs=health_out)
            eval_btn.click(run_eval_stream, outputs=eval_out)

if __name__ == "__main__":
    # Ensure dependencies are available before launch if possible
    # (The user will run this from their terminal as per instructions)
    demo.launch(server_port=settings.gradio_port, server_name="0.0.0.0", inbrowser=True)