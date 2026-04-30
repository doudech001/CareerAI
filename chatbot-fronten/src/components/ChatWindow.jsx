import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

function ChatWindow({
  messages,
  input,
  setInput,
  isBotTyping,
  handleSend,
  messagesEndRef,
  inputRef,
}) {
  return (
    <main className="mainArea">
      <div className="chatScroll">
        <div className="messageStage">
          {messages.length === 0 ? (
            <div className="emptyPrompt">Ask anything...</div>
          ) : (
            <MessageList
              messages={messages}
              isBotTyping={isBotTyping}
              messagesEndRef={messagesEndRef}
            />
          )}
        </div>
      </div>

      <div className="inputOverlay">
        <div className="inputCard">
          <MessageInput
            input={input}
            setInput={setInput}
            isBotTyping={isBotTyping}
            handleSend={handleSend}
            inputRef={inputRef}
          />
          <div className="inputHint">Shift + Enter to add a new line</div>
        </div>
      </div>
    </main>
  );
}

export default ChatWindow;
