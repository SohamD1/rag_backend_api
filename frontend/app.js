const loginView = document.querySelector("#loginView");
const dashboardView = document.querySelector("#dashboardView");
const loginForm = document.querySelector("#loginForm");
const codeInput = document.querySelector("#codeInput");
const loginError = document.querySelector("#loginError");
const logoutButton = document.querySelector("#logoutButton");
const uploadForm = document.querySelector("#uploadForm");
const pdfInput = document.querySelector("#pdfInput");
const sourceUrlInput = document.querySelector("#sourceUrlInput");
const fileName = document.querySelector("#fileName");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const refreshButton = document.querySelector("#refreshButton");
const documentsBody = document.querySelector("#documentsBody");
const emptyState = document.querySelector("#emptyState");

let dashboardToken = "";

function showLogin(message = "") {
  dashboardToken = "";
  dashboardView.hidden = true;
  loginView.hidden = false;
  loginError.textContent = message;
  loginError.hidden = !message;
  codeInput.focus();
}

function showDashboard() {
  loginView.hidden = true;
  dashboardView.hidden = false;
}

async function api(path, options = {}) {
  const headers = {
    ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {}),
  };
  if (dashboardToken && path.startsWith("/api/admin/") && path !== "/api/admin/login") {
    headers.Authorization = `Bearer ${dashboardToken}`;
  }

  const response = await fetch(path, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    showLogin("Enter the current verification code.");
    throw new Error("Unauthorized");
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail =
        typeof payload.detail === "string"
          ? payload.detail
          : payload.detail?.message || JSON.stringify(payload.detail);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || "Request failed");
  }

  return response.json();
}

function iconTrash() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v5M14 11v5" />
    </svg>
  `;
}

function renderDocuments(docs) {
  documentsBody.innerHTML = "";
  emptyState.hidden = docs.length > 0;

  for (const doc of docs) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><div class="pdf-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div></td>
      <td><a class="source-link" href="${escapeAttribute(safeHref(doc.source_url))}" target="_blank" rel="noreferrer">${escapeHtml(doc.source_url)}</a></td>
      <td>${Number(doc.page_count || 0)}</td>
      <td><span class="pill">${escapeHtml(doc.route)} &middot; ${escapeHtml(doc.index_version)}</span></td>
      <td>
        <button class="delete-button" type="button" aria-label="Delete ${escapeAttribute(doc.filename)}" data-doc-id="${escapeAttribute(doc.doc_id)}">
          ${iconTrash()}
        </button>
      </td>
    `;
    documentsBody.appendChild(row);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

function safeHref(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "#";
  } catch {
    return "#";
  }
}

async function loadDocuments() {
  const payload = await api("/api/admin/documents");
  renderDocuments(payload.docs || []);
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;
  const code = codeInput.value.trim();
  try {
    const payload = await api("/api/admin/login", {
      method: "POST",
      body: JSON.stringify({ code }),
    });
    dashboardToken = payload.token || "";
    codeInput.value = "";
    showDashboard();
    await loadDocuments();
  } catch (error) {
    showLogin(error.message === "Unauthorized" ? "Invalid verification code." : error.message);
  }
});

logoutButton.addEventListener("click", async () => {
  showLogin();
});

pdfInput.addEventListener("change", () => {
  const file = pdfInput.files?.[0];
  fileName.textContent = file ? file.name : "Select PDF";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pdfInput.files?.[0];
  if (
    !file ||
    (file.type && file.type !== "application/pdf") ||
    !file.name.toLowerCase().endsWith(".pdf")
  ) {
    uploadStatus.textContent = "Select a PDF file.";
    return;
  }
  if (!sourceUrlInput.value.trim()) {
    uploadStatus.textContent = "Source link is required.";
    return;
  }

  const body = new FormData();
  body.append("file", file);
  body.append("source_url", sourceUrlInput.value.trim());

  uploadButton.disabled = true;
  uploadStatus.textContent = "Ingesting document...";
  try {
    await api("/api/admin/documents", { method: "POST", body });
    uploadForm.reset();
    fileName.textContent = "Select PDF";
    uploadStatus.textContent = "Document added.";
    await loadDocuments();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
});

documentsBody.addEventListener("click", async (event) => {
  const button = event.target.closest(".delete-button");
  if (!button) {
    return;
  }
  const docId = button.dataset.docId;
  if (!docId || !confirm("Delete this document?")) {
    return;
  }
  button.disabled = true;
  try {
    await api(`/api/admin/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    await loadDocuments();
  } catch (error) {
    uploadStatus.textContent = error.message;
    button.disabled = false;
  }
});

refreshButton.addEventListener("click", async () => {
  await loadDocuments();
});

showLogin();
