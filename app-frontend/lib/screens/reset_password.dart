import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';
import 'package:flutter_application_1/utils/color_utils.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class ResetPassword extends StatefulWidget {
  const ResetPassword({super.key});

  @override
  _ResetPasswordState createState() => _ResetPasswordState();
}

class _ResetPasswordState extends State<ResetPassword> {
  final TextEditingController _emailTextController = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Color.fromARGB(255, 51, 85, 255), size: 30),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: SingleChildScrollView(
          child: Container(
            color: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Row(
                mainAxisAlignment: MainAxisAlignment.start, // Aligns the text to the left
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                    "Forgot Password?",
                    style: TextStyle(
                      color: Color.fromARGB(255, 51, 85, 255),
                      fontWeight: FontWeight.w900,
                      fontFamily: "Roboto",
                      fontSize: 30,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    "No worries, we got you!",
                    style: TextStyle(
                      color: Colors.black,
                      fontFamily: "Roboto",
                      fontSize: 14,
                      fontWeight: FontWeight.w600
                    ),
                  )
                    ],
                  )
                ],
              ),
              SizedBox(height: 20),
                 Column(
                  children: <Widget>[
                    const SizedBox(
                      height: 5,
                    ),
                    SvgPicture.asset(
                    'assets/images/forgot_password.svg',  
                    height: 400, 
                    width: 300,  
                  ),
                    const SizedBox(
                      height: 5,
                    ),
                    reusableTextField("Enter Email Id", Icons.person_outline, false,
                        _emailTextController),
                    const SizedBox(
                      height: 20,
                    ),
                    firebaseUIButton(context, "Reset Password", () {
                      FirebaseAuth.instance
                          .sendPasswordResetEmail(email: _emailTextController.text)
                          .then((value) => Navigator.of(context).pop());
                    })
                  ],
                )
              ],
          ))),
    );
  }
}