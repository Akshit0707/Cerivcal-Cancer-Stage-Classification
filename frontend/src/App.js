import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';
import IndiaMap from './IndiaMap';
import Chatbot from './Chatbot';
import { FaShieldAlt, FaHospital, FaChartLine, FaUserMd } from 'react-icons/fa';

const API_URL = 'http://localhost:8000';

const STAGE_DESCRIPTIONS = {
  'Normal': 'Healthy cervical tissue with no abnormalities detected.',
  'CIN1': 'Cervical Intraepithelial Neoplasia Grade 1 - Low-grade dysplasia. Mild abnormalities in cervical cells.',
  'CIN2': 'Cervical Intraepithelial Neoplasia Grade 2 - High-grade dysplasia. Moderate abnormalities requiring medical attention.',
  'CIN3': 'Cervical Intraepithelial Neoplasia Grade 3 - Severe dysplasia. Significant abnormalities requiring immediate medical attention.',
  'Cancer': 'Invasive cervical cancer detected. Requires urgent medical intervention.'
};

// Normalize the API response to a consistent shape regardless of key naming
function normalizeResult(data) {
  if (!data) return null;

  // Support both snake_case and camelCase key variants from the backend
  const predictedClass =
    data.predicted_class ||
    data.prediction ||
    data.class ||
    data.label ||
    null;

  const confidence =
    data.confidence != null ? data.confidence :
    data.confidence_score != null ? data.confidence_score :
    data.score != null ? data.score :
    null;

  const probabilities =
    data.probabilities ||
    data.probs ||
    data.class_probabilities ||
    null;

  return { predicted_class: predictedClass, confidence, probabilities };
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('Please drop a valid image file');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('API raw response:', response.data);

      const normalized = normalizeResult(response.data);
      console.log('Normalized result:', normalized);

      if (!normalized || !normalized.predicted_class) {
        setError(
          'Prediction received but could not read the result. ' +
          'Check console for the raw API response and verify key names match ' +
          '(expected: predicted_class, confidence, probabilities).'
        );
        return;
      }

      setResult(normalized);
    } catch (err) {
      console.error('Prediction error:', err);

      // Give a clear, specific error message
      if (!err.response) {
        // Network error — backend down or CORS
        setError(
          'Could not reach the server at ' + API_URL + '. ' +
          'Make sure the backend is running and CORS is enabled. ' +
          '(Check the browser console for details.)'
        );
      } else {
        setError(
          err.response?.data?.detail ||
          err.response?.data?.message ||
          `Server error ${err.response.status}: ${err.response.statusText}`
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="gov-header">
        <div className="gov-header-top">
          <div className="emblem-section">
            <div className="emblem">🏥</div>
            <div className="header-text">
              <h1>CervixCare</h1>
              <p className="ministry-name-hindi">Advanced AI-Powered Healthcare Solutions</p>
              <p className="ministry-name">Early Detection • Better Treatment • Healthier Lives</p>
            </div>
          </div>
          <div className="header-actions">
            <button className="header-btn" onClick={() => alert('हिंदी version coming soon! | हिंदी संस्करण जल्द ही आ रहा है!')}>हिंदी</button>
            <button className="header-btn" onClick={() => alert('Contact Us:\n\n📞 Helpline: 1800-XXX-XXXX\n✉️ Email: support@cervixcare.com\n🏢 Address: Bangalore, India')}>Contact</button>
          </div>
        </div>
        <div className="gov-header-bottom">
          <h2 className="portal-title">Cervical Cancer Screening & Classification Portal</h2>
          <p className="portal-subtitle">AI-Powered Early Detection System for Better Healthcare Outcomes</p>
        </div>
      </header>

      {/* Quick Stats Banner */}
      <div className="stats-banner">
        <div className="stat-card">
          <FaShieldAlt className="stat-icon" />
          <div className="stat-content">
            <div className="stat-number">90%+</div>
            <div className="stat-label">Preventable with Early Detection</div>
          </div>
        </div>
        <div className="stat-card">
          <FaHospital className="stat-icon" />
          <div className="stat-content">
            <div className="stat-number">5000+</div>
            <div className="stat-label">Government Health Centers</div>
          </div>
        </div>
        <div className="stat-card">
          <FaChartLine className="stat-icon" />
          <div className="stat-content">
            <div className="stat-number">96,922</div>
            <div className="stat-label">Annual Cases in India</div>
          </div>
        </div>
        <div className="stat-card">
          <FaUserMd className="stat-icon" />
          <div className="stat-content">
            <div className="stat-number">Free</div>
            <div className="stat-label">Screening Under NHM</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-wrapper">
        <div className="main-container">
          <div className="upload-section">
            <h2>Upload Cervical Image</h2>

            <div
              className={`upload-area ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="upload-icon">📁</div>
              <div className="upload-text">
                {selectedFile ? selectedFile.name : 'Click to upload or drag and drop'}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="file-input"
              />
            </div>

            {preview && (
              <div className="selected-image">
                <img src={preview} alt="Preview" className="preview-image" />
              </div>
            )}

            <div className="button-group">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                className="gov-button primary"
              >
                {loading ? 'Analyzing Image...' : '🔍 Analyze & Predict Stage'}
              </button>
              {selectedFile && (
                <button onClick={handleReset} className="gov-button secondary">
                  🔄 Clear & Reset
                </button>
              )}
            </div>

            {/* Error display */}
            {error && (
              <div className="error-message" style={{
                marginTop: '16px',
                padding: '12px 16px',
                background: '#fdecea',
                border: '1px solid #f5c6c6',
                borderRadius: '8px',
                color: '#b71c1c',
                fontSize: '14px',
                lineHeight: '1.5'
              }}>
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Government Schemes Info */}
          <div className="info-section">
            <h3>Government Healthcare Schemes</h3>
            <div className="schemes-grid">
              <div className="scheme-card">
                <h4>Ayushman Bharat</h4>
                <p>Free cancer treatment coverage up to ₹5 lakhs per family per year</p>
              </div>
              <div className="scheme-card">
                <h4>National Health Mission</h4>
                <p>Free cervical cancer screening at all government health centers</p>
              </div>
              <div className="scheme-card">
                <h4>HPV Vaccination</h4>
                <p>Government-subsidized HPV vaccines available for eligible age groups</p>
              </div>
            </div>
          </div>

          {/* Results Section */}
          {result && result.predicted_class && (
            <div className="results-section">
              <div className="result-badge">Classification Result</div>
              <h2 className="predicted-class">{result.predicted_class}</h2>

              {result.confidence != null && (
                <div className="confidence-display">
                  <span className="confidence-label">Confidence Score:</span>
                  <span className="confidence-value">
                    {(result.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              )}

              <div className="stage-description">
                <h4>Clinical Description</h4>
                <p>
                  {STAGE_DESCRIPTIONS[result.predicted_class] ||
                    'No description available for this classification.'}
                </p>
              </div>

              <div className="action-recommendations">
                <h4>Recommended Actions</h4>
                <ul>
                  {result.predicted_class === 'Normal' ? (
                    <>
                      <li>Continue regular screening every 3 years</li>
                      <li>Maintain healthy lifestyle</li>
                      <li>Consider HPV vaccination if eligible</li>
                    </>
                  ) : (
                    <>
                      <li>Consult gynecologic oncologist immediately</li>
                      <li>Get confirmatory biopsy and colposcopy</li>
                      <li>Explore treatment options (LEEP, cryotherapy, surgery)</li>
                      <li>Check eligibility for Ayushman Bharat coverage</li>
                    </>
                  )}
                </ul>
              </div>

              {result.probabilities && Object.keys(result.probabilities).length > 0 && (
                <div className="probabilities">
                  <h3>Detailed Stage Probability Analysis</h3>
                  {Object.entries(result.probabilities).map(([stage, prob]) => (
                    <div key={stage} className="probability-bar">
                      <div className="probability-label">
                        <span className="stage-name">{stage}</span>
                        <span className="prob-value">{(prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="bar-container">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${prob * 100}%`,
                            background:
                              prob > 0.5 ? '#d32f2f' :
                              prob > 0.3 ? '#f57c00' : '#1a5490'
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* India Map Section */}
      <IndiaMap />

      {/* Chatbot */}
      <Chatbot />

      {/* Government Footer */}
      <footer className="gov-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Important Links</h4>
            <ul>
              <li><a href="#screening">Find Screening Center</a></li>
              <li><a href="#doctors">Expert Doctors</a></li>
              <li><a href="#schemes">Government Schemes</a></li>
              <li><a href="#awareness">Awareness Programs</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="#guidelines">Clinical Guidelines</a></li>
              <li><a href="#reports">Annual Reports</a></li>
              <li><a href="#research">Research Publications</a></li>
              <li><a href="#training">Training Materials</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Contact Us</h4>
            <ul>
              <li>Helpline: 1800-XXX-XXXX (Toll Free)</li>
              <li>Email: support@cervixcare.com</li>
              <li>CervixCare Healthcare Solutions</li>
              <li>Bangalore, Karnataka, India</li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Disclaimer</h4>
            <p>This AI-based screening tool is for preliminary assessment only. Results must be confirmed by qualified healthcare professionals. Not for clinical diagnosis.</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2026 CervixCare - Advanced Healthcare Solutions | Last Updated: January 3, 2026</p>
          <div className="footer-links">
            <a href="#privacy">Privacy Policy</a>
            <span>|</span>
            <a href="#terms">Terms of Use</a>
            <span>|</span>
            <a href="#accessibility">Accessibility</a>
            <span>|</span>
            <a href="#sitemap">Sitemap</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;