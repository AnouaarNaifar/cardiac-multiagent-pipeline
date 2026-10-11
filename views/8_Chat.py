"""
Streamlit page — Discuss with the assistant (VLM chat over the current case).

Available to every role, with role-specific framing (see ui_chat). It grounds
the conversation in the case's findings and attaches the MRI image when one is
present, so the assistant can actually see it.
"""

import streamlit as st

import ui_common
import ui_chat

ui_common.init_case_store(st)
role = ui_common.current_role(st) or "researcher"

st.title("💬 Discuss with the assistant")
if role == "patient":
    st.error("This assistant explains your results in **plain language**. It does **NOT** give a diagnosis "
             "or prognosis — always consult a qualified doctor.")
else:
    st.caption("A conversational assistant grounded in the current case's findings (and the MRI image when "
               "present). Local multimodal MedGemma.")

contributions = st.session_state.get("contributions", {})
sep = st.session_state.get("sep_decision")
context = ui_chat.build_case_context(contributions, sep)

if not contributions:
    st.info("No case yet — produce findings on a modality page so the assistant has something to discuss. "
            "You can still ask general questions.")

with st.expander("What the assistant sees (case context)"):
    st.text(context)

images = None
mri_b64 = st.session_state.get("mri_ed_b64")
if mri_b64:
    images = [mri_b64]
    st.caption("🖼️ The MRI slice from this case is attached to the conversation.")

st.session_state.setdefault("chat_history", [])

# Replay the conversation.
for m in st.session_state["chat_history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

placeholder = ("Ask about your results…" if role == "patient" else "Ask about the case…")
if prompt := st.chat_input(placeholder):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = ui_chat.answer(role, context, st.session_state["chat_history"][:-1],
                                       prompt, images=images)
            except Exception as exc:  # noqa: BLE001
                reply = f"⚠️ Assistant unavailable — is Ollama running with `medgemma:4b`? ({exc})"
        st.markdown(reply)
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})

if st.session_state["chat_history"]:
    if st.button("🧹 Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()
