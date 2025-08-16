document.addEventListener("DOMContentLoaded", function() {
	const API_CHAT = "http://localhost:5050/chat";   // 필요 시 서버 주소 조정
	const SAVE_URL = "/api/diagnosis-history";

	// 채팅
	const chat = document.getElementById("chat");
	const input = document.getElementById("userInput");
	const sendBtn = document.getElementById("sendBtn");

	// 프로필/히스토리
	const profileSection = document.getElementById("profileSection");
	const historySection = document.getElementById("historySection");
	const historyBody = document.getElementById("history-table-body");
	const historyEmpty = document.getElementById("history-empty");
	const historyCloseBtn = document.getElementById("historyCloseBtn");
	const toggleHistoryBtn = document.getElementById("toggleHistoryBtn"); // ← 제목 우측 '확장' 텍스트 버튼

	const elId = document.getElementById("patient-id");
	const elAge = document.getElementById("patient-age");
	const elGender = document.getElementById("patient-gender");
	const elCond = document.getElementById("patient-conditions");

	// 로그인/회원가입 모달
	const loginBtn = document.getElementById("loginBtn");
	const logoutBtn = document.getElementById("logoutBtn");
	const loginModal = document.getElementById("loginModal");
	const closeLogin = document.getElementById("closeLogin");
	const loginForm = document.getElementById("loginForm");

	const signupBtn = document.getElementById("signupBtn");
	const signupModal = document.getElementById("signupModal");
	const closeSignup = document.getElementById("closeSignup");
	const signupForm = document.getElementById("signupForm");

	let currentPatientId = null;

	// ✅ 히스토리 캐시/상태
	let cachedHistory = null;
	let historyLoadedOnce = false;

	const byId = (id) => document.getElementById(id);

	function escapeHtml(s) { return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
	function fmt(dt) { try { return new Date(dt).toLocaleString("ko-KR", { dateStyle: "medium", timeStyle: "short" }); } catch { return dt ?? ""; } }
	function show(el) { if (el) el.style.display = "block"; }
	function hide(el) { if (el) el.style.display = "none"; }

	// ---- 초기화 유틸 (로그아웃/로그인 시 화면 정리) ----
	function clearChatUI() {
		chat.innerHTML = '<div class="message bot">안녕하세요! 증상을 입력해 주세요.</div>';
	}
	function clearHistoryUI() {
		if (historyBody) historyBody.innerHTML = "";
		if (historyEmpty) historyEmpty.style.display = "none";
		hide(historySection);
		if (toggleHistoryBtn) {
			toggleHistoryBtn.setAttribute("aria-expanded", "false");
			toggleHistoryBtn.textContent = "확장";
		}
	}

	/* ===== 프로필 렌더 ===== */
	function renderPatientProfile(data) {
		elId.textContent = data.id ?? "";
		elAge.textContent = data.age ?? "";
		elGender.textContent = data.gender === "m" ? "남자" : (data.gender === "f" ? "여자" : (data.gender ?? ""));
		elCond.textContent = !data.conditions || data.conditions.trim() === "" ? "없음" : data.conditions;
		show(profileSection);
	}

	/* ===== 히스토리 렌더 ===== */
	function renderHistory(list = []) {
		historyBody.innerHTML = "";
		if (!list || list.length === 0) {
			show(historySection);
			show(historyEmpty);
			return;
		}
		historyEmpty.style.display = "none";
		list.forEach((r) => {
			const tr = document.createElement("tr");
			tr.innerHTML = `
        <td>${fmt(r.chatDate)}</td>
        <td>${escapeHtml(r.symptoms ?? "")}</td>
        <td>${escapeHtml(r.predictedDiagnosis ?? "")}</td>
        <td>${escapeHtml(r.recommendedDepartment ?? "")}</td>
        <td>${escapeHtml(r.additionalInfo ?? "")}</td>`;
			historyBody.appendChild(tr);
		});
		show(historySection);
	}

	// ---- 히스토리 지연 로드(fetch) ----
	async function fetchHistoryOnDemand(patientId) {
		// 1) /patient/{id}/history 우선
		try {
			const res = await fetch(`/patient/${encodeURIComponent(patientId)}/history`, { credentials: "include" });
			if (res.ok) {
				const list = await res.json();
				return Array.isArray(list) ? list : (list?.history || []);
			}
		} catch (_) { }
		// 2) 폴백: /patient/{id}
		try {
			const res2 = await fetch(`/patient/${encodeURIComponent(patientId)}`, { credentials: "include" });
			if (res2.ok) {
				const data2 = await res2.json();
				return data2?.history || [];
			}
		} catch (_) { }
		// 3) 실패 시 빈 배열
		return [];
	}

	async function ensureHistoryLoaded() {
		if (!currentPatientId) return [];
		if (historyLoadedOnce && Array.isArray(cachedHistory)) return cachedHistory;
		const list = await fetchHistoryOnDemand(currentPatientId);
		cachedHistory = list;
		historyLoadedOnce = true;
		return list;
	}

	/* ===== 프로필 불러오기 (로그인 후 호출) ===== */
	async function loadMyProfile(id) {
		// /patient/me 가 있으면 그걸 쓰고, 없으면 /patient/{id}
		const url = id ? `/patient/${encodeURIComponent(id)}` : `/patient/me`;
		const res = await fetch(url, { credentials: "include" });
		if (!res.ok) throw new Error("프로필을 불러오지 못했습니다.");
		const data = await res.json();
		currentPatientId = data.id || id || null;

		// ✅ 기본정보만 먼저 보여주고, 히스토리는 자동으로 펼치지 않음
		renderPatientProfile(data);
		cachedHistory = Array.isArray(data.history) ? data.history : null;
		historyLoadedOnce = Array.isArray(cachedHistory);
		clearHistoryUI(); // 섹션 숨김 + 버튼 '확장'으로
	}

	/* ===== 히스토리 토글(확장/축소) ===== */
	toggleHistoryBtn?.addEventListener("click", async () => {
		if (!currentPatientId) return alert("로그인 후 이용해주세요.");
		const expanded = toggleHistoryBtn.getAttribute("aria-expanded") === "true";
		if (expanded) {
			hide(historySection);
			toggleHistoryBtn.setAttribute("aria-expanded", "false");
			toggleHistoryBtn.textContent = "확장";
			return;
		}
		// 확장
		let list = cachedHistory;
		if (!historyLoadedOnce) {
			toggleHistoryBtn.textContent = "로딩중...";
			list = await ensureHistoryLoaded();
		}
		renderHistory(list || []);
		toggleHistoryBtn.setAttribute("aria-expanded", "true");
		toggleHistoryBtn.textContent = "축소";
	});

	/* ===== 히스토리 닫기(X) ===== */
	historyCloseBtn?.addEventListener("click", () => {
		hide(historySection);
		toggleHistoryBtn?.setAttribute("aria-expanded", "false");
		if (toggleHistoryBtn) toggleHistoryBtn.textContent = "확장";
	});

	/* ===== 모달 오픈/닫기 ===== */
	signupBtn?.addEventListener("click", () => signupModal && (signupModal.style.display = "block"));
	closeSignup?.addEventListener("click", () => signupModal && (signupModal.style.display = "none"));
	loginBtn?.addEventListener("click", () => loginModal && (loginModal.style.display = "block"));
	closeLogin?.addEventListener("click", () => loginModal && (loginModal.style.display = "none"));
	window.addEventListener("click", (e) => {
		if (e.target === signupModal) signupModal.style.display = "none";
		if (e.target === loginModal) loginModal.style.display = "none";
	});

	/* ===== 비밀번호 표시/숨김 ===== */
	function togglePassword(inputEl, iconEl) {
		if (!inputEl || !iconEl) return;
		iconEl.addEventListener("click", () => {
			const toText = inputEl.type === "password";
			inputEl.type = toText ? "text" : "password";
			if (iconEl.classList.contains("fa")) {
				iconEl.classList.toggle("fa-eye");
				iconEl.classList.toggle("fa-eye-slash");
			}
		});
	}
	togglePassword(byId("signupPwd"), byId("togglePwd"));
	togglePassword(byId("signupPwdConfirm"), byId("togglePwdConfirm"));
	togglePassword(byId("loginPassword"), byId("pwToggleLogin"));

	/* ===== 비밀번호 정책 & 실시간 피드백 ===== */
	const pwMsg = byId("pwMatchMsg");
	const passwordRegex = /^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]).{8,20}$/;
	function updatePwFeedback() {
		const p1 = byId("signupPwd")?.value || "";
		const p2 = byId("signupPwdConfirm")?.value || "";
		pwMsg?.classList.remove("ok", "bad"); if (pwMsg) pwMsg.style.display = "none";
		if (!p1 && !p2) return;
		if (!passwordRegex.test(p1)) { if (pwMsg) { pwMsg.textContent = "영문, 숫자, 특수문자를 모두 포함한 8~20자"; pwMsg.classList.add("bad"); pwMsg.style.display = "block"; } return; }
		if (p2 && p1 !== p2) { if (pwMsg) { pwMsg.textContent = "비밀번호가 일치하지 않습니다."; pwMsg.classList.add("bad"); pwMsg.style.display = "block"; } return; }
		if (p2 && p1 === p2) { if (pwMsg) { pwMsg.textContent = "비밀번호가 일치합니다."; pwMsg.classList.add("ok"); pwMsg.style.display = "block"; } }
	}
	byId("signupPwd")?.addEventListener("input", updatePwFeedback);
	byId("signupPwdConfirm")?.addEventListener("input", updatePwFeedback);

	/* ===== 회원가입 ===== */
	signupForm?.addEventListener("submit", async (e) => {
		e.preventDefault();
		const id = byId("signupId")?.value?.trim();
		const age = Number(byId("signupAge")?.value);
		const genderKo = document.querySelector("input[name='signupGender']:checked")?.value;
		const condRaw = byId("signupCondition")?.value?.trim() || "";
		const pwd = byId("signupPwd")?.value || "";
		const pwd2 = byId("signupPwdConfirm")?.value || "";
		if (!id || !age || !genderKo || !pwd || !pwd2) return alert("필수 항목을 모두 입력해주세요.");
		if (!passwordRegex.test(pwd)) return alert("비밀번호는 영문/숫자/특수문자 포함 8~20자");
		if (pwd !== pwd2) return alert("비밀번호가 일치하지 않습니다.");
		const gender = (genderKo === "남") ? "m" : "f";
		const conditions = condRaw === "" ? "없음" : condRaw;

		try {
			const res = await fetch("/patient/register", {
				method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include",
				body: JSON.stringify({ id, age, gender, conditions, password: pwd })
			});
			const txt = await res.text();
			if (!res.ok) throw new Error(txt || "회원가입 실패");
			alert(txt || "회원가입이 완료되었습니다!");
			signupModal.style.display = "none";
			// 필요 시 자동 로그인:
			// const loggedId = await doLogin(id, pwd); await loadMyProfile(loggedId);
		} catch (err) { alert(err.message || "오류가 발생했습니다."); }
	});

	/* ===== 로그인/로그아웃 ===== */
	async function doLogin(id, password) {
		const res = await fetch("/patient/login", {
			method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include",
			body: JSON.stringify({ id, password })
		});
		const text = await res.text();
		if (!res.ok) throw new Error(text || "로그인 실패");
		loginModal.style.display = "none";
		loginBtn.style.display = "none";
		signupBtn.style.display = "none";
		logoutBtn.style.display = "list-item";
		return id;
	}
	loginForm?.addEventListener("submit", async (e) => {
		e.preventDefault();
		const id = byId("loginId")?.value?.trim();
		const pw = byId("loginPassword")?.value || "";
		if (!id || !pw) return alert("아이디/비밀번호를 입력하세요.");
		try {
			const loggedId = await doLogin(id, pw);
			await loadMyProfile(loggedId);
			// 로그인 시 채팅창도 새로 시작하고 싶다면 다음 줄 주석 해제
			// clearChatUI();
		} catch (err) { alert(err.message || "로그인 실패"); }
	});

	logoutBtn?.addEventListener("click", async () => {
		try {
			const res = await fetch("/patient/logout", { method: "POST", credentials: "include" });
			const text = await res.text();
			if (!res.ok) throw new Error(text || "로그아웃 실패");

			// UI 상태 복원
			logoutBtn.style.display = "none";
			loginBtn.style.display = "list-item";
			signupBtn.style.display = "list-item";

			// 세션/뷰 상태 초기화
			currentPatientId = null;
			hide(profileSection);
			clearHistoryUI();   // 과거기록 뷰 초기화
			clearChatUI();      // 채팅창 초기화(인사만 남김)

			alert("로그아웃 되었습니다.");
		} catch (e) {
			alert(e.message || "로그아웃 실패");
		}
	});

	/* ===== 환자 기본정보(전송용) 추출 ===== */
	function getPatientBasicInfoForSend() {
		// 화면에 렌더된 값을 사용 (백엔드/스토리지 접근 불필요)
		const age = elAge.textContent?.trim() || null;

		// 화면에는 '남자'/'여자'로 표기 → 전송은 'm'/'f'로 표준화
		const genderText = (elGender.textContent || "").trim();
		let gender = null;
		if (genderText === "남자") gender = "m";
		else if (genderText === "여자") gender = "f";
		else if (genderText) gender = genderText; // 혹시 다른 값이 있으면 그대로 전달

		// 기저질환: '없음'이면 빈 문자열로 남기거나 '없음' 그대로 전송(정책 선택)
		const conditions = elCond.textContent?.trim() || "";

		return { age, gender, conditions };
	}

	/* ===== 챗봇 ===== */
	input?.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
	sendBtn?.addEventListener("click", sendMessage);

	function addMessage(text, sender) {
		const msg = document.createElement("div");
		msg.classList.add("message", sender);
		msg.textContent = text;
		chat.appendChild(msg);
		chat.scrollTop = chat.scrollHeight;
		return msg;
	}
	function formatSectionHtml(text) {
		const lines = text.split(/\n+/).map((s) => s.trim()).filter(Boolean);
		if (lines.some((l) => /^[-•]/.test(l))) {
			return "<ul>" + lines.map((l) => `<li>${escapeHtml(l.replace(/^[-•]\s?/, ""))}</li>`).join("") + "</ul>";
		}
		return `<p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>`;
	}
	async function showBotAnswer(answer) {
		const sections = String(answer).split(/\n\s*(?=\d+\.\s)/g).filter(Boolean);
		if (sections.length === 0) { addMessage(answer, "bot"); return; }
		for (const sec of sections) {
			const [title, ...rest] = sec.split(/\n/);
			const body = rest.join("\n").trim();
			const box = document.createElement("div");
			box.className = "message bot section";
			box.innerHTML = `<div class="section-title">${escapeHtml(title)}</div>${body ? formatSectionHtml(body) : ""}`;
			chat.appendChild(box);
			chat.scrollTop = chat.scrollHeight;
		}
	}

	function extractDiagnosisParts(result) {
		const response = (result?.response || "").trim();
		const structured = {
			predictedDiagnosis: (result?.predictedDiagnosis || "").trim(),
			diagnosisDefinition: (result?.diagnosisDefinition || "").trim(),
			recommendedDepartment: (result?.recommendedDepartment || "").trim(),
			preventionManagement: (result?.preventionManagement || "").trim(),
			additionalInfo: (result?.additionalInfo || "").trim(),
		};
		if (Object.values(structured).some((v) => v)) return structured;
		if (!response) return structured;

		const blocks = String(response).split(/\n\s*(?=\d+\.\s)/g).map((b) => b.trim()).filter(Boolean).map((b) => {
			const m = b.match(/^\s*(\d+)\.\s*([^\n:：]+)[:：]?\s*/);
			const title = m ? m[2].trim() : "";
			const body = m ? b.slice(m[0].length).trim() : b;
			return { title, body };
		});

		const out = { predictedDiagnosis: "", diagnosisDefinition: "", recommendedDepartment: "", preventionManagement: "", additionalInfo: "" };
		const put = (k, t) => { if (!t) return; out[k] = out[k] ? out[k] + (out[k].endsWith("\n") ? "" : "\n") + t : t; };

		for (const { title, body } of blocks) {
			const t = title || "";
			if (/(예상|예측|가능|병명|감별)/i.test(t)) { put("predictedDiagnosis", body); continue; }
			if (/(정의|개요|원인|특징|증상)/i.test(t)) { put("diagnosisDefinition", body); continue; }
			if (/(추천\s*진료과|진료과|방문\s*진료)/i.test(t)) { put("recommendedDepartment", body); continue; }
			if (/(예방|관리|대처|자가\s*관리|응급)/i.test(t)) { put("preventionManagement", body); continue; }
			if (/(생활|주의|추가|경고|TIP|메모)/i.test(t)) { put("additionalInfo", body); continue; }
			put("additionalInfo", (title ? `${title}\n` : "") + body);
		}
		if (!Object.values(out).some((v) => v)) out.predictedDiagnosis = response;
		return out;
	}

	function prependHistoryRow(r) {
		historyEmpty && (historyEmpty.style.display = "none");
		const tr = document.createElement("tr");
		tr.innerHTML = `
      <td>${fmt(r.chatDate || new Date())}</td>
      <td>${escapeHtml(r.symptoms || "")}</td>
      <td>${escapeHtml(r.predictedDiagnosis || "")}</td>
      <td>${escapeHtml(r.recommendedDepartment || "")}</td>
      <td>${escapeHtml(r.additionalInfo || "")}</td>`;
		historyBody.firstChild ? historyBody.insertBefore(tr, historyBody.firstChild) : historyBody.appendChild(tr);
	}

	async function saveDiagnosisIfNeeded(symptomsText, chatResult) {
		// 로그인(세션)된 사용자만 저장
		if (!currentPatientId) return;
		const p = extractDiagnosisParts(chatResult);
		const payload = {
			patientId: currentPatientId,
			symptoms: symptomsText,
			predictedDiagnosis: p.predictedDiagnosis || "",
			diagnosisDefinition: p.diagnosisDefinition || "",
			recommendedDepartment: p.recommendedDepartment || "",
			preventionManagement: p.preventionManagement || "",
			additionalInfo: p.additionalInfo || "",
		};
		const res = await fetch(SAVE_URL, {
			method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include",
			body: JSON.stringify(payload)
		});
		const text = await res.text();
		if (!res.ok) {
			if (res.status === 401) alert("로그인이 필요합니다.");
			else if (res.status === 403) alert("본인 계정의 기록만 저장할 수 있습니다.");
			console.error("SAVE_FAIL", res.status, text); return;
		}
		let saved = null; try { saved = JSON.parse(text); } catch { }
		// 캐시에도 반영
		const row = saved || { ...payload, chatDate: new Date().toISOString() };
		cachedHistory = Array.isArray(cachedHistory) ? [row, ...cachedHistory] : [row];
		historyLoadedOnce = true;
		// 화면에 열려 있으면 즉시 반영
		if (historySection && historySection.style.display !== "none") {
			prependHistoryRow(row);
		}
	}

	function sendMessage() {
		const message = input.value.trim();
		if (!message) return;

		const userMsg = document.createElement("div");
		userMsg.classList.add("message", "user");
		userMsg.textContent = message;
		chat.appendChild(userMsg);
		chat.scrollTop = chat.scrollHeight;
		input.value = "";

		const loadingMsg = document.createElement("div");
		loadingMsg.classList.add("message", "bot");
		loadingMsg.textContent = "답변 생성 중...";
		chat.appendChild(loadingMsg);

		// ✅ 환자 기본정보에서 age, gender, conditions만 전송
		const patient = getPatientBasicInfoForSend();

		fetch(API_CHAT, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			// 서버 호환성 유지: 기존 message 필드 + 신규 symptom/patient/history 동시 전송
			body: JSON.stringify({
				message,                 // 기존 서버가 쓰던 필드(호환 목적)
				symptom: message,        // 새 필드: 증상
				patient,                 // { age, gender(m/f), conditions }
				history: null            // 과거기록은 아직 미전송
			})
		})
			.then((r) => r.json())
			.then(async (data) => {
				loadingMsg.remove();
				if (data.response) { await showBotAnswer(data.response); }
				else { addMessage("응답이 없습니다.", "bot"); }
				try { await saveDiagnosisIfNeeded(message, data); } catch (e) { console.warn(e); }
			})
			.catch((err) => {
				loadingMsg.remove();
				addMessage("서버와 통신 중 오류가 발생했습니다.", "bot");
				console.error(err);
			});
	}

	/* ===== 메뉴(햄버거) ===== */
	const menuToggle = document.getElementById("menuToggle");
	const sideMenu = document.getElementById("sideMenu");
	const menuOverlay = document.getElementById("menuOverlay");

	function setMenuHiddenPosition() {
		const menuWidth = sideMenu?.offsetWidth || 240;
		if (sideMenu) sideMenu.style.left = `-${menuWidth + 10}px`;
	}
	setMenuHiddenPosition();
	menuToggle?.addEventListener("click", () => {
		sideMenu?.classList.add("open");
		if (sideMenu) sideMenu.style.left = "0";
		menuOverlay?.classList.add("show");
	});
	menuOverlay?.addEventListener("click", () => {
		setMenuHiddenPosition();
		sideMenu?.classList.remove("open");
		menuOverlay?.classList.remove("show");
	});
	document.querySelectorAll("#sideMenu a").forEach((link) => {
		link.addEventListener("click", () => {
			setMenuHiddenPosition();
			sideMenu?.classList.remove("open");
			menuOverlay?.classList.remove("show");
		});
	});

	/* ===== 글씨 크기/다크모드 ===== */
	let currentFontSize = 17; const minFontSize = 13, maxFontSize = 32;
	function setMsgFontSize(px) { document.documentElement.style.setProperty("--msg-font-size", px + "px"); }
	byId("fontIncrease")?.addEventListener("click", () => { if (currentFontSize < maxFontSize) { currentFontSize += 2; setMsgFontSize(currentFontSize); } });
	byId("fontDecrease")?.addEventListener("click", () => { if (currentFontSize > minFontSize) { currentFontSize -= 2; setMsgFontSize(currentFontSize); } });
	setMsgFontSize(currentFontSize);

	const darkModeBtn = byId("darkModeBtn"); const body = document.body;
	if (localStorage.getItem("darkMode") === "on") { body.classList.add("dark"); if (darkModeBtn) darkModeBtn.textContent = "☀️"; }
	darkModeBtn?.addEventListener("click", function() {
		if (body.classList.toggle("dark")) { darkModeBtn.textContent = "☀️"; localStorage.setItem("darkMode", "on"); }
		else { darkModeBtn.textContent = "🌙"; localStorage.setItem("darkMode", "off"); }
	});
});
