import axios from 'axios'
import { useEffect, useRef, useState } from 'react'

const API_ROOT = 'http://localhost:5000'
const SUGGESTED_QUESTIONS = [
  'What are the symptoms of lumpy skin disease?',
  'How can mastitis be prevented?',
  'What are the signs of foot-and-mouth disease?',
  'How can I maintain healthy cattle?',
  'When should I contact a veterinarian?',
]

function renderInline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${keyPrefix}-${index}`}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

function renderAnswer(text) {
  const elements = []
  let bulletItems = []
  const flushBullets = () => {
    if (bulletItems.length) {
      elements.push(<ul key={`list-${elements.length}`}>{bulletItems}</ul>)
      bulletItems = []
    }
  }

  String(text || '').split(/\r?\n/).forEach((line, index) => {
    const trimmed = line.trim()
    if (!trimmed) {
      flushBullets()
      return
    }
    const heading = trimmed.match(/^#{1,3}\s+(.+)$/)
    const bullet = trimmed.match(/^[-*•]\s+(.+)$/)
    if (heading) {
      flushBullets()
      elements.push(<h4 key={`heading-${index}`}>{renderInline(heading[1], `heading-${index}`)}</h4>)
    } else if (bullet) {
      bulletItems.push(<li key={`bullet-${index}`}>{renderInline(bullet[1], `bullet-${index}`)}</li>)
    } else {
      flushBullets()
      elements.push(<p key={`paragraph-${index}`}>{renderInline(trimmed, `paragraph-${index}`)}</p>)
    }
  })
  flushBullets()
  return elements
}

function DiseaseKnowledgeChat({ reviewImage = null, predictionContext = null }) {
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'How can I help your cattle today?',
      intro: 'Ask about cattle diseases, symptoms, prevention, or care described in the veterinary references. I will show the sources behind each response.',
      answer_mode: 'welcome',
      sources: [],
    },
  ])
  const [knowledgeStatus, setKnowledgeStatus] = useState(null)
  const [connectionError, setConnectionError] = useState(false)
  const [loading, setLoading] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [adminMessage, setAdminMessage] = useState('')
  const [attachedImage, setAttachedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const messageEndRef = useRef(null)
  const imageInputRef = useRef(null)

  useEffect(() => {
    if (!attachedImage) {
      setImagePreview(null)
      return undefined
    }
    const previewUrl = URL.createObjectURL(attachedImage)
    setImagePreview(previewUrl)
    return () => URL.revokeObjectURL(previewUrl)
  }, [attachedImage])

  useEffect(() => {
    axios.get(`${API_ROOT}/knowledge/status`)
      .then((response) => {
        setKnowledgeStatus(response.data)
        setConnectionError(false)
      })
      .catch(() => setConnectionError(true))
  }, [])

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  const sendQuestion = async (question, image = attachedImage) => {
    const normalizedQuestion = question.trim()
    if (!normalizedQuestion || loading) return

    const history = messages
      .filter((message) => message.id !== 'welcome' && ['user', 'assistant'].includes(message.role))
      .slice(-8)
      .map((message) => ({ role: message.role, content: message.content }))
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: normalizedQuestion,
      attachment: image?.name,
    }
    setMessages((current) => [...current, userMessage])
    setDraft('')
    setLoading(true)
    setConnectionError(false)

    try {
      const formData = new FormData()
      formData.append('question', normalizedQuestion)
      formData.append('history', JSON.stringify(history))
      if (image) {
        formData.append('image', image)
        if (image === reviewImage && predictionContext) {
          const context = Object.fromEntries(
            ['disease', 'stage', 'confidence', 'clinical_support_score', 'evidence_strength', 'confidence_level', 'clinical_summary', 'available_evidence']
              .filter((key) => predictionContext[key] != null)
              .map((key) => [key, predictionContext[key]]),
          )
          formData.append('prediction_context', JSON.stringify(context))
        }
      }
      const response = await axios.post(`${API_ROOT}/knowledge/ask`, formData)
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.data.answer,
        answer_mode: response.data.answer_mode,
        status: response.data.status,
        generation_warning: response.data.generation_warning,
        safety_note: response.data.safety_note,
        image_observations: response.data.image_observations,
        image_analysis_status: response.data.image_analysis_status,
        sources: response.data.sources || [],
      }])
    } catch (error) {
      const backendIsUnreachable = !error.response
      setConnectionError(!error.response)
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: backendIsUnreachable
          ? 'I cannot connect to the assistant right now. Please check that the Flask backend is running, then try again.'
          : error.response?.data?.error?.includes('index has not been built')
            ? 'The reference library is not ready yet. Open Reference library settings and refresh it, then try again.'
            : error.response?.data?.error || 'I could not process that question. Please try rephrasing it.',
        answer_mode: 'error',
        sources: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    void sendQuestion(draft)
  }

  const handleImageChange = (event) => {
    const file = event.target.files?.[0]
    if (file) setAttachedImage(file)
    event.target.value = ''
  }

  const handleInputKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void sendQuestion(draft)
    }
  }

  const rebuildLibrary = async () => {
    setReindexing(true)
    setAdminMessage('Refreshing the reference library…')
    try {
      const response = await axios.post(`${API_ROOT}/knowledge/reindex`)
      setKnowledgeStatus((current) => ({ ...current, ...response.data, indexed: response.data.built }))
      setAdminMessage(response.data.built
        ? `Reference library updated with ${response.data.chunk_count} passages.`
        : response.data.error || 'No readable text was found in the PDFs.')
      setConnectionError(false)
    } catch (error) {
      setAdminMessage(error.response?.data?.error || 'Could not refresh the reference library.')
    } finally {
      setReindexing(false)
    }
  }

  return (
    <section id="knowledge" className="knowledge-assistant" aria-labelledby="knowledge-title">
      <header className="assistant-header">
        <div className="assistant-identity">
          <span className="assistant-brand-mark" aria-hidden="true">KR</span>
          <div>
            <h2 id="knowledge-title">KrishiRakshak AI Assistant</h2>
            <p>Your cattle health knowledge companion</p>
          </div>
        </div>
      </header>

      <div className="assistant-chat-log" role="log" aria-label="Conversation with KrishiRakshak AI Assistant" aria-live="polite">
        {connectionError && (
          <div className="assistant-connection-note" role="status">
            The assistant service is not reachable. Check that the Flask backend is running at localhost:5000.
          </div>
        )}
        {messages.map((message) => (
          <article className={`assistant-message-row ${message.role === 'user' ? 'from-user' : 'from-assistant'}`} key={message.id}>
            {message.role === 'assistant' && <span className="chat-avatar" aria-hidden="true">KR</span>}
            <div className="assistant-message-column">
              {message.role === 'assistant' && (
                <div className="assistant-message-label">KrishiRakshak AI Assistant</div>
              )}
                <div className={`assistant-bubble ${message.role === 'user' ? 'user-bubble' : 'ai-bubble'} ${['error', 'llm_not_configured'].includes(message.answer_mode) ? 'error-bubble' : ''}`}>
                {message.intro && <p className="assistant-welcome-copy">{message.intro}</p>}
                <div className="assistant-answer-content">{renderAnswer(message.content)}</div>
                  {message.attachment && <p className="assistant-attachment-name">Image attached: {message.attachment}</p>}
                  {message.image_observations && (
                    <div className="assistant-image-observation">
                      <strong>Visual model suggestion, not a diagnosis</strong>
                      <p>{message.image_observations}</p>
                    </div>
                  )}
                  {message.image_analysis_status === 'unusable' && (
                    <div className="assistant-image-observation">
                      <strong>Image analysis limitation</strong>
                      <p>The image did not yield a reliable visible observation for this question.</p>
                    </div>
                  )}
              </div>
              {message.sources?.length > 0 && (
                <details className="assistant-sources">
                  <summary>Sources <span>{message.sources.length}</span></summary>
                  <div className="assistant-source-list">
                    {message.sources.map((source) => (
                      <details className="assistant-source-item" key={`${message.id}-${source.id}`}>
                        <summary>{source.filename}<span>Page {source.page}</span></summary>
                        <blockquote>{source.excerpt}</blockquote>
                      </details>
                    ))}
                  </div>
                </details>
              )}
            </div>
            {message.role === 'user' && <span className="user-avatar" aria-hidden="true">You</span>}
          </article>
        ))}
        {messages.length === 1 && !loading && (
          <div className="assistant-suggestions" aria-label="Suggested questions">
            {SUGGESTED_QUESTIONS.map((question) => (
              <button key={question} type="button" onClick={() => void sendQuestion(question, null)}>
                <span aria-hidden="true">↗</span>{question}
              </button>
            ))}
          </div>
        )}
        {loading && (
          <div className="assistant-message-row from-assistant" role="status">
            <span className="chat-avatar" aria-hidden="true">KR</span>
            <div className="assistant-message-column">
              <div className="assistant-message-label">KrishiRakshak AI Assistant</div>
              <div className="assistant-bubble ai-bubble typing-indicator">KrishiRakshak is thinking<span /><span /><span /></div>
            </div>
          </div>
        )}
        <div ref={messageEndRef} />
      </div>

      <form className="assistant-composer" onSubmit={handleSubmit}>
        <div className="assistant-attachment-tools">
          <input
            ref={imageInputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            onChange={handleImageChange}
            tabIndex={-1}
          />
          <button type="button" className="assistant-attach-button" onClick={() => imageInputRef.current?.click()} disabled={loading}>
            {attachedImage ? 'Replace image' : 'Attach image'}
          </button>
          {reviewImage && attachedImage !== reviewImage && (
            <button type="button" className="assistant-attach-button" onClick={() => setAttachedImage(reviewImage)} disabled={loading}>
              Use current review image
            </button>
          )}
          {attachedImage && (
            <button type="button" className="assistant-remove-image" onClick={() => setAttachedImage(null)} disabled={loading}>
              Remove image
            </button>
          )}
        </div>
        {attachedImage && imagePreview && (
          <div className="assistant-image-preview">
            <img src={imagePreview} alt="Selected image for the veterinary question" />
            <span>{attachedImage.name}</span>
          </div>
        )}
        <label className="visually-hidden" htmlFor="assistant-message">Message KrishiRakshak AI Assistant</label>
        <textarea
          id="assistant-message"
          rows="2"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="Ask about cattle health…"
          disabled={loading}
        />
        <button type="submit" disabled={!draft.trim() || loading} title="Send message">
          {loading ? 'Sending…' : 'Send'}
        </button>
        <p className="assistant-safety-note">For serious symptoms or suspected contagious disease, contact a qualified veterinarian and follow local animal-health reporting guidance.</p>
      </form>

      <details className="assistant-diagnostics">
        <summary>Reference library settings</summary>
        <div className="assistant-diagnostics-content">
          <p>{knowledgeStatus?.documents?.length || 0} veterinary PDFs · {knowledgeStatus?.chunk_count || 0} indexed passages</p>
          <button type="button" onClick={rebuildLibrary} disabled={reindexing}>
            {reindexing ? 'Refreshing references…' : 'Refresh reference library'}
          </button>
          {adminMessage && <p role="status">{adminMessage}</p>}
        </div>
      </details>
    </section>
  )
}

export default DiseaseKnowledgeChat