package com.medbot.controller;

import com.google.zxing.BarcodeFormat;
import com.google.zxing.WriterException;
import com.google.zxing.client.j2se.MatrixToImageWriter;
import com.google.zxing.qrcode.QRCodeWriter;
import com.google.zxing.common.BitMatrix;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.ui.Model;

import java.io.IOException;
import java.io.ByteArrayOutputStream;

@Controller
public class QrController {

	@Value("${app.base-url}")
	private String appBaseUrl;

	// ✅ /qrcode 엔드포인트는 이제 URL을 직접 받아 QR을 생성합니다.
	@GetMapping(value = "/qrcode", produces = MediaType.IMAGE_PNG_VALUE)
	@ResponseBody
	public byte[] qrcode(@RequestParam String url) throws WriterException, IOException {
		int size = 320;
		QRCodeWriter writer = new QRCodeWriter();
		BitMatrix matrix = writer.encode(url, BarcodeFormat.QR_CODE, size, size);

		try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
			MatrixToImageWriter.writeToStream(matrix, "PNG", baos);
			return baos.toByteArray();
		}
	}

	// ✅ /qr 엔드포인트는 최종 URL을 완성하여 뷰로 전달합니다.
	@GetMapping("/qr")
	public String qrLanding(@RequestParam(defaultValue = "/home") String target, Model model) {
		String fullUrl = appBaseUrl + (target.startsWith("/") ? target : ("/" + target));
		String qrImgSrc = "/qrcode?url=" + fullUrl; // ✅ QR 생성 시 사용할 URL을 파라미터로 전달

		model.addAttribute("fullUrl", fullUrl);
		model.addAttribute("qrImgSrc", qrImgSrc);
		model.addAttribute("targetPath", target);
		return "qr";
	}
}