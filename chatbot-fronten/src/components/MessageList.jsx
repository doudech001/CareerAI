import MessageBubble from "./MessageBubble";

function MessageList({ messages, isBotTyping, messagesEndRef }) {
  return (
    <div className="messages">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isBotTyping && (
  <div className="message bot typingBubble">
    <span className="typingDot"></span>
    <span className="typingDot"></span>
    <span className="typingDot"></span>
  </div>
)}

     

      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;