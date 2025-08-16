package com.medbot.controller;

import com.medbot.domain.DiagnosisHistory;
import com.medbot.domain.Patient;
import com.medbot.repository.DiagnosisHistoryRepository;
import com.medbot.repository.PatientRepository;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.*;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
public class PatientController {

	private final PatientRepository patientRepo;
	private final DiagnosisHistoryRepository historyRepo;
	private final BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();

	public PatientController(PatientRepository patientRepo, DiagnosisHistoryRepository historyRepo) {
		this.patientRepo = patientRepo;
		this.historyRepo = historyRepo;
	}

	/** 회원가입: 엔티티 + write-only password 사용 */
	@PostMapping("/patient/register")
	public ResponseEntity<String> register(@RequestBody Patient req) {
		if (req.getId() == null || req.getAge() == null || req.getGender() == null || req.getPassword() == null) {
			return ResponseEntity.badRequest().body("필수 항목 누락");
		}
		if (patientRepo.existsById(req.getId())) {
			return ResponseEntity.status(HttpStatus.CONFLICT).body("이미 존재하는 아이디입니다.");
		}

		String conditions = (req.getConditions() == null || req.getConditions().trim().isEmpty()) ? "없음"
				: req.getConditions().trim();

		Patient p = new Patient();
		p.setId(req.getId());
		p.setAge(req.getAge());
		p.setGender(req.getGender());
		p.setConditions(conditions);
		p.setPasswordHash(encoder.encode(req.getPassword())); // 응답에 노출 안 됨

		patientRepo.save(p);
		return ResponseEntity.ok("회원가입이 완료되었습니다.");
	}

	/** 로그인: 간단 Map으로 입력 */
	@PostMapping("/patient/login")
	public ResponseEntity<String> login(@RequestBody Map<String, String> body, HttpSession session) {
		String id = body.get("id");
		String password = body.get("password");
		if (id == null || password == null)
			return ResponseEntity.badRequest().body("아이디/비밀번호를 입력하세요.");

		Optional<Patient> opt = patientRepo.findById(id);
		if (opt.isEmpty())
			return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("아이디 또는 비밀번호가 올바르지 않습니다.");

		Patient p = opt.get();
		if (p.getPasswordHash() != null && encoder.matches(password, p.getPasswordHash())) {
			session.setAttribute("LOGIN_ID", p.getId());
			return ResponseEntity.ok("로그인 성공");
		}
		return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("아이디 또는 비밀번호가 올바르지 않습니다.");
	}

	@PostMapping("/patient/logout")
	public ResponseEntity<String> logout(HttpSession session) {
		session.invalidate();
		return ResponseEntity.ok("로그아웃 되었습니다.");
	}

	/** 환자 기본 + 히스토리 묶어서 반환(별도 DTO 없이 Map) */
	@GetMapping("/patient/{id}")
	public ResponseEntity<?> getPatient(@PathVariable String id) {
		return patientRepo.findById(id).<ResponseEntity<?>>map(p -> {
			List<DiagnosisHistory> history = historyRepo.findByPatientIdOrderByChatDateDesc(p.getId());
			Map<String, Object> out = new LinkedHashMap<>();
			out.put("id", p.getId());
			out.put("age", p.getAge());
			out.put("gender", p.getGender());
			out.put("conditions", p.getConditions());
			out.put("history", history); // 엔티티 그대로(민감정보 없음)
			return ResponseEntity.ok(out);
		}).orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND).build());
	}
}
