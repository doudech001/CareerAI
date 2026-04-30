function MessageBubble({ message }) {
  const isUser = message.sender === "user";

  return (
    <div className={`messageRow ${isUser ? "userRow" : "botRow"}`}>
      <div className={`messageCard ${isUser ? "userCard" : "botCard"}`}>
        <span className={`messageLabel ${isUser ? "messageLabelUser" : "messageLabelBot"}`}>
          {isUser ? "User" : "Assistant v1.0"}
        </span>
        <div className="messageText">{message.text}</div>
      </div>
    </div>
  );
}

export default MessageBubble;
