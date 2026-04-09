import React, { useState, useRef, useEffect } from 'react';
import { FaComments, FaTimes, FaPaperPlane } from 'react-icons/fa';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: 'Namaste! I am your Government Healthcare Assistant. How can I help you today?',
      options: [
        'Find Diagnostic Centers',
        'Find Expert Doctors',
        'Learn About Cervical Cancer',
        'Understand Test Results'
      ]
    }
  ]);
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const diagnosticCenters = [
    { name: 'AIIMS New Delhi', location: 'New Delhi', phone: '011-26588500', speciality: 'Cancer Screening' },
    { name: 'Tata Memorial Hospital', location: 'Mumbai, Maharashtra', phone: '022-24177000', speciality: 'Cancer Treatment' },
    { name: 'CMC Vellore', location: 'Vellore, Tamil Nadu', phone: '0416-228-1000', speciality: 'Women\'s Health' },
    { name: 'Rajiv Gandhi Cancer Institute', location: 'Delhi', phone: '011-4705-8000', speciality: 'Oncology' },
    { name: 'Apollo Hospitals', location: 'Multiple Locations', phone: '1860-500-1066', speciality: 'Comprehensive Care' },
  ];

  const expertDoctors = [
    { name: 'Dr. Sunesh Kumar', speciality: 'Gynecologic Oncologist', hospital: 'AIIMS Delhi', experience: '20+ years' },
    { name: 'Dr. Amita Maheshwari', speciality: 'Cytopathologist', hospital: 'Tata Memorial Hospital', experience: '15+ years' },
    { name: 'Dr. Neerja Bhatla', speciality: 'Gynecologic Oncology', hospital: 'AIIMS Delhi', experience: '25+ years' },
    { name: 'Dr. Ravi Mehrotra', speciality: 'Cancer Screening Expert', hospital: 'National Institute of Cancer Prevention', experience: '30+ years' },
  ];

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const userMessage = { type: 'user', text: inputText };
    setMessages(prev => [...prev, userMessage]);
    
    setTimeout(() => {
      const response = generateResponse(inputText.toLowerCase());
      setMessages(prev => [...prev, response]);
    }, 500);

    setInputText('');
  };

  const handleOptionClick = (option) => {
    const userMessage = { type: 'user', text: option };
    setMessages(prev => [...prev, userMessage]);

    setTimeout(() => {
      const response = generateResponse(option.toLowerCase());
      setMessages(prev => [...prev, response]);
    }, 500);
  };

  const generateResponse = (input) => {
    if (input.includes('diagnostic') || input.includes('center') || input.includes('test')) {
      return {
        type: 'bot',
        text: 'Here are some premier diagnostic centers for cervical cancer screening:',
        centers: diagnosticCenters,
        options: ['Find Expert Doctors', 'Learn About Cervical Cancer', 'Main Menu']
      };
    } else if (input.includes('doctor') || input.includes('expert') || input.includes('specialist')) {
      return {
        type: 'bot',
        text: 'Here are leading cervical cancer specialists in India:',
        doctors: expertDoctors,
        options: ['Find Diagnostic Centers', 'Learn About Cervical Cancer', 'Main Menu']
      };
    } else if (input.includes('learn') || input.includes('cervical cancer') || input.includes('information')) {
      return {
        type: 'bot',
        text: `Cervical Cancer Information:

• Cervical cancer is caused primarily by HPV (Human Papillomavirus)
• Regular screening can detect precancerous changes early
• Pap smear test recommended every 3 years for women 21-65
• HPV vaccination available for prevention (ages 9-45)
• Early detection leads to 90%+ survival rate

Government Schemes:
• Free screening under National Health Mission
• Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat) covers treatment
• State-specific cancer screening programs`,
        options: ['Find Diagnostic Centers', 'Find Expert Doctors', 'Main Menu']
      };
    } else if (input.includes('result') || input.includes('understand') || input.includes('stage')) {
      return {
        type: 'bot',
        text: `Understanding Test Results:

• Normal: No abnormalities detected - Regular screening recommended
• CIN1: Low-grade changes - Usually monitored, may resolve naturally
• CIN2/CIN3: High-grade changes - Requires treatment (LEEP, cryotherapy)
• Cancer: Requires immediate oncologist consultation and treatment plan

Next Steps:
1. Consult with a gynecologic oncologist
2. Get confirmatory tests (colposcopy, biopsy)
3. Discuss treatment options
4. Explore government healthcare schemes for support`,
        options: ['Find Diagnostic Centers', 'Find Expert Doctors', 'Main Menu']
      };
    } else if (input.includes('main menu') || input.includes('start over')) {
      return {
        type: 'bot',
        text: 'How can I help you today?',
        options: [
          'Find Diagnostic Centers',
          'Find Expert Doctors',
          'Learn About Cervical Cancer',
          'Understand Test Results'
        ]
      };
    } else {
      return {
        type: 'bot',
        text: 'I can help you with:\n\n• Finding diagnostic centers near you\n• Locating expert doctors\n• Information about cervical cancer\n• Understanding test results\n\nPlease select an option or type your question.',
        options: [
          'Find Diagnostic Centers',
          'Find Expert Doctors',
          'Learn About Cervical Cancer',
          'Understand Test Results'
        ]
      };
    }
  };

  return (
    <>
      {!isOpen && (
        <button className="chatbot-toggle" onClick={() => setIsOpen(true)}>
          <FaComments className="chat-icon" />
          <span className="chat-badge">Healthcare Assistant</span>
        </button>
      )}

      {isOpen && (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <div className="chatbot-header-content">
              <FaComments className="header-icon" />
              <div>
                <h3>Government Healthcare Assistant</h3>
                <span className="header-subtitle">Ministry of Health & Family Welfare</span>
              </div>
            </div>
            <button className="close-button" onClick={() => setIsOpen(false)}>
              <FaTimes />
            </button>
          </div>

          <div className="chatbot-messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.type}`}>
                <div className="message-content">
                  <p style={{ whiteSpace: 'pre-line' }}>{message.text}</p>
                  
                  {message.centers && (
                    <div className="info-cards">
                      {message.centers.map((center, idx) => (
                        <div key={idx} className="info-card">
                          <h4>🏥 {center.name}</h4>
                          <p><strong>Location:</strong> {center.location}</p>
                          <p><strong>Phone:</strong> {center.phone}</p>
                          <p><strong>Speciality:</strong> {center.speciality}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {message.doctors && (
                    <div className="info-cards">
                      {message.doctors.map((doctor, idx) => (
                        <div key={idx} className="info-card">
                          <h4>👨‍⚕️ {doctor.name}</h4>
                          <p><strong>Speciality:</strong> {doctor.speciality}</p>
                          <p><strong>Hospital:</strong> {doctor.hospital}</p>
                          <p><strong>Experience:</strong> {doctor.experience}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {message.options && (
                    <div className="message-options">
                      {message.options.map((option, idx) => (
                        <button
                          key={idx}
                          className="option-button"
                          onClick={() => handleOptionClick(option)}
                        >
                          {option}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message..."
            />
            <button onClick={handleSendMessage} className="send-button">
              <FaPaperPlane />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;
