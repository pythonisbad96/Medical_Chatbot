package com.medbot.model;

//PatientRepository.java
import java.util.List;
import java.sql.Timestamp;
import java.util.ArrayList;

//진단 기록을 담을 DTO(Data Transfer Object) 클래스
public class DiagnosisHistory {
	private String symptoms;
	private String diagnosisResult;
	private Timestamp chatDate;

	// 생성자, getter, setter 등 구현...
	public DiagnosisHistory(String symptoms, String diagnosisResult, Timestamp chatDate) {
		this.symptoms = symptoms;
		this.diagnosisResult = diagnosisResult;
		this.chatDate = chatDate;
	}

	public String getSymptoms() {
		return symptoms;
	}

	public String getDiagnosisResult() {
		return diagnosisResult;
	}

	public Timestamp getChatDate() {
		return chatDate;
	}

	@Override
	public String toString() {
		return "증상: " + symptoms + ", 진단: " + diagnosisResult + ", 날짜: " + chatDate;
	}
}