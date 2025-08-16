package com.medbot.domain;

public class ChatRequest {
	private String message;
	private String patientId;

	public ChatRequest() {
	}

	public ChatRequest(String message, String patientId) {
		this.message = message;
		this.patientId = patientId;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public String getPatientId() {
		return patientId;
	}

	public void setPatientId(String patientId) {
		this.patientId = patientId;
	}
}
