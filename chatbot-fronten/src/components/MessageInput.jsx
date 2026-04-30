function MessageInput({ input, setInput, isBotTyping, handleSend, inputRef }) {
  return (
    <div className="inputArea">
      <textarea
        ref={inputRef}
        value={input}
        disabled={isBotTyping}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSend();
          }
        }}
        placeholder="Type your message..."
        rows={1}
        autoFocus
      />
      <button className="sendButton" onClick={handleSend} disabled={!input.trim() || isBotTyping}>
        <span className="material-symbols-outlined" data-weight="fill">send</span>
      </button>
    </div>
  );
}

export default MessageInput;
