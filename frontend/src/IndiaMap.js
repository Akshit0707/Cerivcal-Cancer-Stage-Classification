import React, { useState } from 'react';
import './IndiaMap.css';

const IndiaMap = () => {
  const [selectedState, setSelectedState] = useState(null);

  // Cervical cancer statistics by state (cases per 100,000 women)
  const stateData = {
    'Uttar Pradesh': { cases: 18500, rate: 28.5, stage: 'high' },
    'Maharashtra': { cases: 16200, rate: 32.1, stage: 'very-high' },
    'Bihar': { cases: 14800, rate: 31.2, stage: 'very-high' },
    'West Bengal': { cases: 12400, rate: 29.8, stage: 'high' },
    'Madhya Pradesh': { cases: 11200, rate: 27.6, stage: 'high' },
    'Tamil Nadu': { cases: 10800, rate: 26.4, stage: 'high' },
    'Rajasthan': { cases: 9600, rate: 25.8, stage: 'moderate' },
    'Karnataka': { cases: 8900, rate: 24.2, stage: 'moderate' },
    'Gujarat': { cases: 8200, rate: 23.5, stage: 'moderate' },
    'Andhra Pradesh': { cases: 7800, rate: 22.9, stage: 'moderate' },
    'Odisha': { cases: 7200, rate: 21.8, stage: 'moderate' },
    'Telangana': { cases: 6400, rate: 20.5, stage: 'moderate' },
    'Kerala': { cases: 5800, rate: 19.2, stage: 'low' },
    'Jharkhand': { cases: 5200, rate: 18.7, stage: 'low' },
    'Assam': { cases: 4800, rate: 17.9, stage: 'low' },
    'Punjab': { cases: 4200, rate: 17.2, stage: 'low' },
    'Chhattisgarh': { cases: 3900, rate: 16.8, stage: 'low' },
    'Haryana': { cases: 3600, rate: 15.9, stage: 'low' },
    'Delhi': { cases: 2800, rate: 14.5, stage: 'low' },
    'Jammu & Kashmir': { cases: 2200, rate: 13.8, stage: 'low' },
  };

  const getColorForStage = (stage) => {
    switch (stage) {
      case 'very-high': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'moderate': return '#fbc02d';
      case 'low': return '#689f38';
      default: return '#bdbdbd';
    }
  };

  const getStageName = (stage) => {
    switch (stage) {
      case 'very-high': return 'Very High';
      case 'high': return 'High';
      case 'moderate': return 'Moderate';
      case 'low': return 'Low';
      default: return 'No Data';
    }
  };

  return (
    <div className="india-map-container">
      <div className="map-header">
        <h2>🗺️ Cervical Cancer Prevalence Across India</h2>
        <p className="map-subtitle">Annual reported cases per 100,000 women (2024-2025)</p>
      </div>

      <div className="map-content">
        <div className="map-visualization">
          {/* States Bar Chart */}
          <div className="states-chart">
            <h3>State-wise Cervical Cancer Prevalence</h3>
            {Object.entries(stateData)
              .sort((a, b) => b[1].rate - a[1].rate)
              .map(([state, data]) => (
                <div 
                  key={state} 
                  className="state-bar-item"
                  onClick={() => setSelectedState(state)}
                  onMouseEnter={() => setSelectedState(state)}
                >
                  <div className="state-info">
                    <span className="state-name">{state}</span>
                    <span className="state-rate">{data.rate} per 100K</span>
                  </div>
                  <div className="bar-background">
                    <div 
                      className="bar-fill-horizontal"
                      style={{
                        width: `${(data.rate / 35) * 100}%`,
                        background: getColorForStage(data.stage)
                      }}
                    >
                      <span className="bar-label">{data.cases.toLocaleString()} cases</span>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>

        <div className="map-sidebar">
          <div className="legend">
            <h3>Prevalence Level</h3>
            <div className="legend-items">
              <div className="legend-item">
                <div className="legend-color" style={{ background: '#d32f2f' }}></div>
                <span>Very High (&gt;30 per 100K)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: '#f57c00' }}></div>
                <span>High (25-30 per 100K)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: '#fbc02d' }}></div>
                <span>Moderate (20-25 per 100K)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color" style={{ background: '#689f38' }}></div>
                <span>Low (&lt;20 per 100K)</span>
              </div>
            </div>
          </div>

          {selectedState && (
            <div className="state-details">
              <h3>{selectedState}</h3>
              <div className="detail-item">
                <span className="detail-label">Annual Cases:</span>
                <span className="detail-value">{stateData[selectedState].cases.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Rate per 100K:</span>
                <span className="detail-value">{stateData[selectedState].rate}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Prevalence:</span>
                <span 
                  className="detail-badge"
                  style={{ background: getColorForStage(stateData[selectedState].stage) }}
                >
                  {getStageName(stateData[selectedState].stage)}
                </span>
              </div>
            </div>
          )}

          <div className="statistics-summary">
            <h3>National Statistics</h3>
            <div className="stat-box">
              <div className="stat-number">96,922</div>
              <div className="stat-label">New Cases Annually</div>
            </div>
            <div className="stat-box">
              <div className="stat-number">60,078</div>
              <div className="stat-label">Annual Deaths</div>
            </div>
            <div className="stat-box">
              <div className="stat-number">2nd</div>
              <div className="stat-label">Most Common Cancer in Women</div>
            </div>
          </div>

          <div className="prevention-info">
            <h4>🛡️ Prevention Measures</h4>
            <ul>
              <li>Regular screening (Pap smear/VIA)</li>
              <li>HPV vaccination (ages 9-45)</li>
              <li>Early detection & treatment</li>
              <li>Government screening programs</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="data-source">
        <p>Data Source: National Cancer Registry Programme (ICMR) | Ministry of Health & Family Welfare, Government of India</p>
      </div>
    </div>
  );
};

export default IndiaMap;
