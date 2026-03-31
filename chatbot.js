/**
 * ════════════════════════════════════════════════════════════
 * CHATBOT UI - JavaScript Logic
 * Học viện Kỹ thuật và Công nghệ An ninh
 * 
 * Features:
 *  - Session management (localStorage)
 *  - File upload with doc_id tracking
 *  - Parallel session requests (non-blocking)
 *  - Search across conversations
 *  - Auto-generated conversation titles
 * ════════════════════════════════════════════════════════════
 */

(() => {
    'use strict';

    // ─── Configuration ───────────────────────────────────────
    const API_BASE = window.location.origin;
    const STORAGE_KEY = 'academy_chatbot_conversations';
    const MAX_TITLE_LENGTH = 55;
    const SEARCH_DEBOUNCE_MS = 300;
    const AUTO_MESSAGE_FOR_FILES = (fileNames) =>
        `[Tài liệu đã tải lên: ${fileNames.join(', ')}]`;

    // ─── DOM References ──────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const DOM = {
        sidebar: $('#sidebar'),
        sidebarCollapseBtn: $('#sidebar-collapse-btn'),
        sidebarOpenBtn: $('#sidebar-open-btn'),
        sidebarOverlay: $('#sidebar-overlay'),
        newChatBtn: $('#new-chat-btn'),
        searchInput: $('#search-input'),
        searchClearBtn: $('#search-clear-btn'),
        searchResults: $('#search-results'),
        conversationList: $('#conversation-list'),
        mainContent: $('#main-content'),
        conversationTitle: $('#conversation-title'),
        docCountBadge: $('#doc-count-badge'),
        docCountText: $('#doc-count-text'),
        chatArea: $('#chat-area'),
        welcomeScreen: $('#welcome-screen'),
        messagesContainer: $('#messages-container'),
        attachmentsBar: $('#attachments-bar'),
        attachmentsList: $('#attachments-list'),
        attachBtn: $('#attach-btn'),
        fileInput: $('#file-input'),
        chatInput: $('#chat-input'),
        sendBtn: $('#send-btn'),
        contextMenu: $('#context-menu'),
        ctxRename: $('#ctx-rename'),
        ctxDelete: $('#ctx-delete'),
    };

    // ─── State ───────────────────────────────────────────────
    let conversations = [];          // Array of conversation objects
    let activeConvId = null;         // Current active conversation ID
    let pendingFiles = [];           // Files selected but not uploaded yet
    let activeRequests = new Map();  // Map<convId, AbortController> for parallel sessions
    let contextMenuTargetId = null;  // Conversation ID for context menu

    // ─── Marked.js Config ────────────────────────────────────
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return code;
            }
        });
    }

    // ═════════════════════════════════════════════════════════
    // STORAGE
    // ═════════════════════════════════════════════════════════
    function loadConversations() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            conversations = raw ? JSON.parse(raw) : [];
        } catch (e) {
            console.error('Failed to load conversations:', e);
            conversations = [];
        }
    }

    function saveConversations() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
        } catch (e) {
            console.error('Failed to save conversations:', e);
        }
    }

    function getConversation(id) {
        return conversations.find(c => c.id === id);
    }

    // ═════════════════════════════════════════════════════════
    // CONVERSATION MANAGEMENT
    // ═════════════════════════════════════════════════════════
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }

    function createConversation() {
        const conv = {
            id: generateUUID(),
            title: 'Cuộc hội thoại mới',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            doc_ids: [],
            messages: [],
        };
        conversations.unshift(conv);
        saveConversations();
        return conv;
    }

    function generateTitle(message) {
        // Take first meaningful content, strip markdown, truncate
        let title = message
            .replace(/[#*_~`>]/g, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/\n+/g, ' ')
            .trim();
        if (title.length > MAX_TITLE_LENGTH) {
            title = title.substring(0, MAX_TITLE_LENGTH).trim() + '...';
        }
        return title || 'Cuộc hội thoại mới';
    }

    function deleteConversation(id) {
        conversations = conversations.filter(c => c.id !== id);
        saveConversations();
        if (activeConvId === id) {
            if (conversations.length > 0) {
                switchToConversation(conversations[0].id);
            } else {
                activeConvId = null;
                showWelcomeScreen();
                DOM.conversationTitle.textContent = 'Cuộc hội thoại mới';
                updateDocCountBadge([]);
            }
        }
        renderConversationList();
    }

    function renameConversation(id, newTitle) {
        const conv = getConversation(id);
        if (conv) {
            conv.title = newTitle.trim() || conv.title;
            conv.updated_at = new Date().toISOString();
            saveConversations();
            renderConversationList();
            if (id === activeConvId) {
                DOM.conversationTitle.textContent = conv.title;
            }
        }
    }

    // ═════════════════════════════════════════════════════════
    // UI RENDERING
    // ═════════════════════════════════════════════════════════

    // ── Conversation List (grouped by time) ──────────────────
    function getTimeGroup(dateStr) {
        const date = new Date(dateStr);
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today.getTime() - 86400000);
        const week = new Date(today.getTime() - 7 * 86400000);
        const month = new Date(today.getTime() - 30 * 86400000);

        if (date >= today) return 'Hôm nay';
        if (date >= yesterday) return 'Hôm qua';
        if (date >= week) return '7 ngày trước';
        if (date >= month) return '30 ngày trước';
        return 'Cũ hơn';
    }

    function renderConversationList() {
        const container = DOM.conversationList;
        container.innerHTML = '';

        if (conversations.length === 0) {
            container.innerHTML = `
                <div style="padding: 24px 16px; text-align: center; color: var(--text-sidebar-muted); font-size: 0.8rem;">
                    Chưa có cuộc hội thoại nào
                </div>`;
            return;
        }

        // Group by time
        const groups = {};
        const groupOrder = ['Hôm nay', 'Hôm qua', '7 ngày trước', '30 ngày trước', 'Cũ hơn'];
        conversations.forEach(conv => {
            const group = getTimeGroup(conv.updated_at);
            if (!groups[group]) groups[group] = [];
            groups[group].push(conv);
        });

        groupOrder.forEach(groupName => {
            if (!groups[groupName]) return;
            const label = document.createElement('div');
            label.className = 'conv-group-label';
            label.textContent = groupName;
            container.appendChild(label);

            groups[groupName].forEach(conv => {
                const item = document.createElement('div');
                item.className = `conv-item${conv.id === activeConvId ? ' active' : ''}`;
                item.dataset.convId = conv.id;
                item.innerHTML = `
                    <div class="conv-item-icon">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                    </div>
                    <div class="conv-item-content">
                        <div class="conv-item-title">${escapeHTML(conv.title)}</div>
                    </div>
                    <button class="conv-item-menu-btn" data-conv-id="${conv.id}" title="Tùy chọn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>
                        </svg>
                    </button>
                `;

                // Click to switch conversation
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.conv-item-menu-btn')) return;
                    switchToConversation(conv.id);
                });

                container.appendChild(item);
            });
        });

        // Attach menu button listeners
        container.querySelectorAll('.conv-item-menu-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                showContextMenu(e, btn.dataset.convId);
            });
        });
    }

    // ── Messages ─────────────────────────────────────────────
    function renderMessages(conv) {
        const container = DOM.messagesContainer;
        container.innerHTML = '';

        if (!conv || conv.messages.length === 0) {
            showWelcomeScreen();
            return;
        }

        DOM.welcomeScreen.style.display = 'none';
        container.style.display = 'block';

        conv.messages.forEach(msg => {
            appendMessageToDOM(msg, false);
        });

        scrollToBottom();
    }

    function appendMessageToDOM(msg, animate = true) {
        const container = DOM.messagesContainer;
        DOM.welcomeScreen.style.display = 'none';
        container.style.display = 'block';

        const div = document.createElement('div');
        div.className = `message ${msg.role}`;
        if (!animate) div.style.animation = 'none';

        const avatarContent = msg.role === 'user'
            ? '<span>Bạn</span>'
            : `<img src="/static/Logo_H%E1%BB%8Dc_vi%E1%BB%87n_K%E1%BB%B9_thu%E1%BA%ADt_v%C3%A0_C%C3%B4ng_ngh%E1%BB%87_An_ninh_n%C4%83m_2025.png" alt="AI">`;

        const senderName = msg.role === 'user' ? 'Bạn' : 'Trợ lý AI';

        // Render file chips if present
        let filesHTML = '';
        if (msg.files && msg.files.length > 0) {
            const fileChips = msg.files.map(f => {
                const ext = (f || '').split('.').pop().toLowerCase();
                let iconClass = 'default';
                if (['doc', 'docx'].includes(ext)) iconClass = 'word';
                if (ext === 'pdf') iconClass = 'pdf';

                if (msg.role === 'user') {
                    // Uploaded files
                    const iconHTML = msg.is_uploading ? `
                        <svg class="file-spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10" stroke-dasharray="28 28" stroke-dashoffset="14" stroke-linecap="round"></circle>
                        </svg>
                    ` : `
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                        </svg>
                    `;
                    return `
                    <div class="message-file-chip ${msg.is_uploading ? 'uploading' : ''}">
                        <div class="message-file-icon-box ${iconClass}">
                            ${iconHTML}
                        </div>
                        <div class="message-file-info">
                            <span class="message-file-title" title="${escapeHTML(f)}">${escapeHTML(f)}</span>
                            <span class="message-file-type">Document</span>
                        </div>
                    </div>`;
                } else {
                    // Downloadable files
                    return `<a href="${API_BASE}/download/${encodeURIComponent(f)}" class="download-file-btn" target="_blank">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        ${escapeHTML(f)}
                    </a>`;
                }
            }).join('');
            
            if (msg.role === 'user') {
                filesHTML = `<div class="message-files">${fileChips}</div>`;
            } else {
                filesHTML = `<div class="download-files">${fileChips}</div>`;
            }
        }

        // Render content
        let contentHTML = '';
        if (msg.role === 'assistant' && typeof marked !== 'undefined') {
            contentHTML = marked.parse(msg.content || '');
        } else {
            contentHTML = escapeHTML(msg.content || '');
        }

        // Hide text bubble if content is empty OR if it is the auto-generated neutral file string
        const safeContent = String(msg.content || '').trim();
        const isNeutral = safeContent.startsWith('[Tài liệu đã tải lên:');
        const showContent = (safeContent.length > 0 && !isNeutral);

        div.innerHTML = `
            <div class="message-avatar">${avatarContent}</div>
            <div class="message-body">
                <div class="message-sender">${senderName}</div>
                ${msg.role === 'user' ? filesHTML : ''}
                ${showContent ? `<div class="message-content">${contentHTML}</div>` : ''}
                ${msg.role === 'assistant' ? filesHTML : ''}
            </div>
        `;

        container.appendChild(div);

        // Add Excel export buttons to tables in assistant messages
        if (msg.role === 'assistant') {
            addExcelExportButtons(div);
        }

        if (animate) scrollToBottom();
    }

    function showTypingIndicator() {
        const container = DOM.messagesContainer;
        const existing = container.querySelector('.typing-message');
        if (existing) return;

        const div = document.createElement('div');
        div.className = 'message assistant typing-message';
        div.innerHTML = `
            <div class="message-avatar">
                <img src="/static/Logo_H%E1%BB%8Dc_vi%E1%BB%87n_K%E1%BB%B9_thu%E1%BA%ADt_v%C3%A0_C%C3%B4ng_ngh%E1%BB%87_An_ninh_n%C4%83m_2025.png" alt="AI">
            </div>
            <div class="message-body">
                <div class="message-sender">Trợ lý AI</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(div);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const el = DOM.messagesContainer.querySelector('.typing-message');
        if (el) el.remove();
    }

    function showWelcomeScreen() {
        DOM.welcomeScreen.style.display = 'flex';
        DOM.messagesContainer.style.display = 'none';
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            DOM.chatArea.scrollTop = DOM.chatArea.scrollHeight;
        });
    }

    // ── Doc Count Badge ──────────────────────────────────────
    function updateDocCountBadge(docIds) {
        if (docIds && docIds.length > 0) {
            DOM.docCountBadge.style.display = 'flex';
            DOM.docCountText.textContent = `${docIds.length} tài liệu`;
        } else {
            DOM.docCountBadge.style.display = 'none';
        }
    }

    // ── Context Menu ─────────────────────────────────────────
    function showContextMenu(event, convId) {
        contextMenuTargetId = convId;
        const menu = DOM.contextMenu;
        menu.style.display = 'block';
        
        // Position near click
        const x = Math.min(event.clientX, window.innerWidth - 180);
        const y = Math.min(event.clientY, window.innerHeight - 100);
        menu.style.left = x + 'px';
        menu.style.top = y + 'px';
    }

    function hideContextMenu() {
        DOM.contextMenu.style.display = 'none';
        contextMenuTargetId = null;
    }

    // ═════════════════════════════════════════════════════════
    // CONVERSATION SWITCHING
    // ═════════════════════════════════════════════════════════
    function switchToConversation(convId) {
        activeConvId = convId;
        const conv = getConversation(convId);
        if (!conv) return;

        // Update title
        DOM.conversationTitle.textContent = conv.title;
        
        // Update doc badge
        updateDocCountBadge(conv.doc_ids);

        // Clear pending files
        clearPendingFiles();

        // Render messages
        renderMessages(conv);
        renderConversationList();

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            closeSidebar();
        }
    }

    // ═════════════════════════════════════════════════════════
    // FILE HANDLING
    // ═════════════════════════════════════════════════════════
    function handleFileSelect(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        files.forEach(file => {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!['docx', 'pdf', 'doc'].includes(ext)) {
                showToast(`Không hỗ trợ: ${file.name}. Chỉ chấp nhận .docx, .pdf, .doc`);
                return;
            }
            // Avoid duplicates
            if (!pendingFiles.find(f => f.name === file.name)) {
                pendingFiles.push(file);
            }
        });

        renderPendingFiles();
        updateSendBtnState();
        event.target.value = ''; // Reset input
    }

    function renderPendingFiles() {
        const bar = DOM.attachmentsBar;
        const list = DOM.attachmentsList;

        if (pendingFiles.length === 0) {
            bar.style.display = 'none';
            return;
        }

        bar.style.display = 'block';
        list.innerHTML = pendingFiles.map((file, idx) => {
            const ext = (file.name || '').split('.').pop().toLowerCase();
            let iconClass = 'default';
            if (['doc', 'docx'].includes(ext)) iconClass = 'word';
            if (ext === 'pdf') iconClass = 'pdf';

            return `
            <div class="attachment-chip" data-idx="${idx}">
                <div class="attachment-chip-icon ${iconClass}">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                </div>
                <div class="attachment-chip-info">
                    <span class="attachment-chip-title" title="${escapeHTML(file.name)}">${escapeHTML(file.name)}</span>
                    <span class="attachment-chip-type">Document</span>
                </div>
                <button class="chip-remove" data-idx="${idx}" title="Xóa">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            `;
        }).join('');

        // Remove handlers
        list.querySelectorAll('.chip-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const idx = parseInt(btn.dataset.idx);
                pendingFiles.splice(idx, 1);
                renderPendingFiles();
                updateSendBtnState();
            });
        });
    }

    function clearPendingFiles() {
        pendingFiles = [];
        renderPendingFiles();
        updateSendBtnState();
    }

    /**
     * Upload files in parallel and return their doc_ids
     */
    async function uploadFiles(files, abortSignal) {
        const uploadPromises = files.map(async (file) => {
            const formData = new FormData();
            formData.append('file', file);

            const resp = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                body: formData,
                signal: abortSignal,
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `Upload failed: ${file.name}`);
            }

            const data = await resp.json();
            return { file_name: file.name, doc_id: data.doc_id };
        });

        return Promise.all(uploadPromises);
    }

    // ═════════════════════════════════════════════════════════
    // SEND MESSAGE
    // ═════════════════════════════════════════════════════════
    async function sendMessage() {
        const message = DOM.chatInput.value.trim();
        const filesToUpload = [...pendingFiles];

        // Handle case: no message AND no files
        if (!message && filesToUpload.length === 0) return;

        // Ensure we have an active conversation
        let conv = getConversation(activeConvId);
        if (!conv) {
            conv = createConversation();
            activeConvId = conv.id;
        }

        // Disable input during processing
        DOM.chatInput.value = '';
        DOM.chatInput.style.height = 'auto';
        updateSendBtnState();

        // Create AbortController for this session (parallel support)
        const abortController = new AbortController();
        activeRequests.set(conv.id, abortController);

        // 1. Determine the actual message to send
        let apiMessage = message;
        let displayMessage = message;
        let isFileOnly = !message && filesToUpload.length > 0;

        if (isFileOnly) {
            // User sent files without text → send neutral notification to API
            const fileNames = filesToUpload.map(f => f.name);
            apiMessage = AUTO_MESSAGE_FOR_FILES(fileNames);
            displayMessage = ''; // Don't show any text in the bubble
        }

        // 2. Add user message to conversation FIRST
        const userMsg = {
            role: 'user',
            content: displayMessage,
            _apiMessage: apiMessage, // Internal: actual message sent to API (not displayed)
            timestamp: new Date().toISOString(),
            files: filesToUpload.length > 0 ? filesToUpload.map(f => f.name) : undefined,
            is_uploading: filesToUpload.length > 0,
            is_file_only: isFileOnly,
        };
        conv.messages.push(userMsg);

        // 3. Auto-generate title from first real text message (skip file-only uploads)
        if (!isFileOnly && conv.title === 'Cuộc hội thoại mới' && displayMessage) {
            conv.title = generateTitle(displayMessage);
            DOM.conversationTitle.textContent = conv.title;
        }

        conv.updated_at = new Date().toISOString();
        saveConversations();
        renderConversationList();

        // 4. Show user message in UI immediately
        if (activeConvId === conv.id) {
            // If this is the first message, clear any stale content from a previous conversation
            if (conv.messages.length === 1) {
                DOM.messagesContainer.innerHTML = '';
            }
            // Only show user message if there's content or files to display
            if (displayMessage || (userMsg.files && userMsg.files.length > 0)) {
                appendMessageToDOM(userMsg);
            }
        }

        try {
            // 5. Upload files (if any)
            let uploadedFileNames = [];
            let newDocIds = [];

            if (filesToUpload.length > 0) {
                clearPendingFiles();

                const results = await uploadFiles(filesToUpload, abortController.signal);
                uploadedFileNames = results.map(r => r.file_name);
                newDocIds = results.map(r => r.doc_id);

                // Add doc_ids to conversation
                newDocIds.forEach(id => {
                    if (!conv.doc_ids.includes(id)) {
                        conv.doc_ids.push(id);
                    }
                });
                updateDocCountBadge(conv.doc_ids);

                // Mark as uploaded and refresh messages to hide spinner
                userMsg.is_uploading = false;
                saveConversations();
                if (activeConvId === conv.id) {
                    renderMessages(conv);
                }
            }

            // Show typing indicator after upload finishes
            if (activeConvId === conv.id) {
                showTypingIndicator();
            }

            // 6. Call chat API
            const chatResp = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: apiMessage,
                    session_id: conv.id,
                    doc_ids: conv.doc_ids,
                }),
                signal: abortController.signal,
            });

            if (!chatResp.ok) {
                const errData = await chatResp.json().catch(() => ({}));
                throw new Error(errData.detail || 'Lỗi khi gửi tin nhắn');
            }

            const chatData = await chatResp.json();

            // 7. Add assistant response
            const assistantMsg = {
                role: 'assistant',
                content: chatData.response || 'Không có phản hồi.',
                timestamp: new Date().toISOString(),
                files: chatData.files && chatData.files.length > 0 ? chatData.files : undefined,
            };
            conv.messages.push(assistantMsg);
            conv.updated_at = new Date().toISOString();
            saveConversations();

            // 8. Update UI if still viewing this conversation
            if (activeConvId === conv.id) {
                removeTypingIndicator();
                appendMessageToDOM(assistantMsg);
            }

        } catch (err) {
            if (err.name === 'AbortError') {
                console.log(`Request aborted for session ${conv.id}`);
                return;
            }

            console.error('Send message error:', err);

            // Remove typing indicator
            if (activeConvId === conv.id) {
                removeTypingIndicator();

                // Show error as assistant message
                const errorMsg = {
                    role: 'assistant',
                    content: `⚠️ **Lỗi:** ${err.message}`,
                    timestamp: new Date().toISOString(),
                };
                conv.messages.push(errorMsg);
                saveConversations();
                appendMessageToDOM(errorMsg);
            }
        } finally {
            activeRequests.delete(conv.id);
        }
    }

    // ═════════════════════════════════════════════════════════
    // SEARCH
    // ═════════════════════════════════════════════════════════
    let searchTimeout = null;

    function handleSearch() {
        const query = DOM.searchInput.value.trim().toLowerCase();
        
        DOM.searchClearBtn.style.display = query ? 'flex' : 'none';

        if (!query) {
            DOM.searchResults.style.display = 'none';
            return;
        }

        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => performSearch(query), SEARCH_DEBOUNCE_MS);
    }

    function performSearch(query) {
        const results = [];

        conversations.forEach(conv => {
            // Search in title
            const titleMatch = conv.title.toLowerCase().includes(query);

            // Search in messages
            let snippetMatch = null;
            for (const msg of conv.messages) {
                const content = msg.content.toLowerCase();
                const idx = content.indexOf(query);
                if (idx !== -1) {
                    // Extract snippet around match
                    const start = Math.max(0, idx - 30);
                    const end = Math.min(msg.content.length, idx + query.length + 50);
                    let snippet = (start > 0 ? '...' : '') + 
                                  msg.content.substring(start, end) + 
                                  (end < msg.content.length ? '...' : '');
                    snippetMatch = snippet;
                    break;
                }
            }

            if (titleMatch || snippetMatch) {
                results.push({
                    conv,
                    titleMatch,
                    snippet: snippetMatch || conv.messages[0]?.content?.substring(0, 80) || '',
                });
            }
        });

        renderSearchResults(results, query);
    }

    function renderSearchResults(results, query) {
        const container = DOM.searchResults;

        if (results.length === 0) {
            container.innerHTML = '<div class="search-no-results">Không tìm thấy kết quả</div>';
            container.style.display = 'block';
            return;
        }

        container.innerHTML = results.map(r => `
            <div class="search-result-item" data-conv-id="${r.conv.id}">
                <div class="search-result-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    ${highlightText(escapeHTML(r.conv.title), query)}
                </div>
                <div class="search-result-snippet">
                    ${highlightText(escapeHTML(r.snippet), query)}
                </div>
            </div>
        `).join('');

        container.style.display = 'block';

        // Click handlers
        container.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                switchToConversation(item.dataset.convId);
                DOM.searchInput.value = '';
                DOM.searchClearBtn.style.display = 'none';
                container.style.display = 'none';
            });
        });
    }

    function highlightText(text, query) {
        if (!query) return text;
        const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<span class="search-highlight">$1</span>');
    }

    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // ═════════════════════════════════════════════════════════
    // SIDEBAR TOGGLE
    // ═════════════════════════════════════════════════════════
    function toggleSidebar() {
        if (window.innerWidth <= 768) {
            DOM.sidebar.classList.toggle('open');
            DOM.sidebarOverlay.classList.toggle('open');
        } else {
            DOM.sidebar.classList.toggle('collapsed');
        }
    }

    function closeSidebar() {
        if (window.innerWidth <= 768) {
            DOM.sidebar.classList.remove('open');
            DOM.sidebarOverlay.classList.remove('open');
        }
    }

    // ═════════════════════════════════════════════════════════
    // INPUT HANDLING
    // ═════════════════════════════════════════════════════════
    function updateSendBtnState() {
        const hasText = DOM.chatInput.value.trim().length > 0;
        const hasFiles = pendingFiles.length > 0;
        DOM.sendBtn.disabled = !hasText && !hasFiles;
    }

    function autoResizeInput() {
        const el = DOM.chatInput;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    }

    // ═════════════════════════════════════════════════════════
    // EXCEL EXPORT
    // ═════════════════════════════════════════════════════════
    function addExcelExportButtons(messageDiv) {
        const tables = messageDiv.querySelectorAll('.message-content table');
        if (tables.length === 0) return;

        tables.forEach((table, idx) => {
            // Skip if already wrapped
            if (table.parentElement.classList.contains('table-export-wrapper')) return;

            // Create wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'table-export-wrapper';

            // Create export button
            const btn = document.createElement('button');
            btn.className = 'export-excel-btn';
            btn.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Xuất Excel
            `;
            btn.addEventListener('click', () => exportTableToExcel(table, idx));

            // Wrap table
            table.parentNode.insertBefore(wrapper, table);
            wrapper.appendChild(btn);
            wrapper.appendChild(table);
        });
    }

    function exportTableToExcel(tableEl, index) {
        if (typeof XLSX === 'undefined') {
            showToast('Lỗi: Thư viện xuất Excel chưa được tải.');
            return;
        }

        try {
            const wb = XLSX.utils.table_to_book(tableEl, { sheet: 'Dữ liệu' });

            // Generate filename from conversation title + table index
            const conv = getConversation(activeConvId);
            const title = conv ? conv.title.replace(/[^\w\sÀ-ỹ]/gi, '').trim().substring(0, 40) : 'bang';
            const timestamp = new Date().toISOString().slice(0, 10);
            const filename = `${title}_bang${index + 1}_${timestamp}.xlsx`;

            XLSX.writeFile(wb, filename);
            showToast('✅ Đã xuất file Excel thành công!');
        } catch (err) {
            console.error('Excel export error:', err);
            showToast('⚠️ Lỗi khi xuất Excel: ' + err.message);
        }
    }

    // ═════════════════════════════════════════════════════════
    // UTILITIES
    // ═════════════════════════════════════════════════════════
    function escapeHTML(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function showToast(message) {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            background: #1e293b; color: #f8fafc; padding: 12px 24px; 
            border-radius: 8px; font-size: 0.85rem; z-index: 9999;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
            animation: msgFadeIn 0.3s ease-out;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ═════════════════════════════════════════════════════════
    // EVENT LISTENERS
    // ═════════════════════════════════════════════════════════
    function initEventListeners() {
        // Sidebar toggle
        DOM.sidebarCollapseBtn.addEventListener('click', toggleSidebar);
        DOM.sidebarOpenBtn.addEventListener('click', toggleSidebar);
        DOM.sidebarOverlay.addEventListener('click', closeSidebar);

        // New chat
        DOM.newChatBtn.addEventListener('click', () => {
            const conv = createConversation();
            activeConvId = conv.id;
            DOM.conversationTitle.textContent = conv.title;
            updateDocCountBadge([]);
            clearPendingFiles();
            showWelcomeScreen();
            renderConversationList();
            DOM.chatInput.focus();
            closeSidebar();
        });

        // Search
        DOM.searchInput.addEventListener('input', handleSearch);
        DOM.searchClearBtn.addEventListener('click', () => {
            DOM.searchInput.value = '';
            DOM.searchClearBtn.style.display = 'none';
            DOM.searchResults.style.display = 'none';
        });

        // Close search results on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-container')) {
                DOM.searchResults.style.display = 'none';
            }
        });

        // File attach
        DOM.attachBtn.addEventListener('click', () => DOM.fileInput.click());
        DOM.fileInput.addEventListener('change', handleFileSelect);

        // Chat input
        DOM.chatInput.addEventListener('input', () => {
            autoResizeInput();
            updateSendBtnState();
        });

        DOM.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!DOM.sendBtn.disabled) sendMessage();
            }
        });

        // Send button
        DOM.sendBtn.addEventListener('click', () => {
            if (!DOM.sendBtn.disabled) sendMessage();
        });

        // Context menu
        DOM.ctxRename.addEventListener('click', () => {
            const targetId = contextMenuTargetId;
            hideContextMenu();
            if (!targetId) return;
            
            const convItem = DOM.conversationList.querySelector(
                `.conv-item[data-conv-id="${targetId}"]`
            );
            if (!convItem) return;

            const titleEl = convItem.querySelector('.conv-item-title');
            const conv = getConversation(targetId);
            const origTitle = conv ? conv.title : '';

            // Replace with input
            const input = document.createElement('input');
            input.className = 'conv-item-rename-input';
            input.value = origTitle;
            titleEl.replaceWith(input);
            input.focus();
            input.select();

            const finishRename = () => {
                renameConversation(targetId, input.value);
            };

            input.addEventListener('blur', finishRename);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    input.blur();
                }
                if (e.key === 'Escape') {
                    input.value = origTitle;
                    input.blur();
                }
            });
        });

        DOM.ctxDelete.addEventListener('click', () => {
            const targetId = contextMenuTargetId;
            hideContextMenu();
            if (targetId) {
                deleteConversation(targetId);
            }
        });

        // Close context menu on outside click
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.context-menu') && !e.target.closest('.conv-item-menu-btn')) {
                hideContextMenu();
            }
        });

        // Suggestion chips
        $$('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                DOM.chatInput.value = chip.dataset.prompt;
                autoResizeInput();
                updateSendBtnState();
                DOM.chatInput.focus();
            });
        });

        // Drag & Drop files on chat area
        DOM.chatArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.chatArea.style.background = 'rgba(14, 116, 144, 0.04)';
        });
        DOM.chatArea.addEventListener('dragleave', () => {
            DOM.chatArea.style.background = '';
        });
        DOM.chatArea.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.chatArea.style.background = '';
            const files = Array.from(e.dataTransfer.files);
            files.forEach(file => {
                const ext = file.name.split('.').pop().toLowerCase();
                if (['docx', 'pdf', 'doc'].includes(ext)) {
                    if (!pendingFiles.find(f => f.name === file.name)) {
                        pendingFiles.push(file);
                    }
                }
            });
            renderPendingFiles();
            updateSendBtnState();
        });
    }

    // ═════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═════════════════════════════════════════════════════════
    function init() {
        loadConversations();
        renderConversationList();

        // Show welcome screen by default (new chat)
        activeConvId = null;
        showWelcomeScreen();
        DOM.conversationTitle.textContent = 'Cuộc hội thoại mới';
        updateDocCountBadge([]);

        initEventListeners();

        console.log('🤖 Chatbot UI initialized');
    }

    // Start
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
