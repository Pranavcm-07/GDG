// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';
// import 'home_screen.dart';
// import 'package:flutter_application_1/utils/color_utils.dart';
// import 'package:flutter/material.dart';

// class SignUpScreen extends StatefulWidget {
//   const SignUpScreen({super.key});

//   @override
//   _SignUpScreenState createState() => _SignUpScreenState();
// }

// class _SignUpScreenState extends State<SignUpScreen> {
//   final TextEditingController _passwordTextController = TextEditingController();
//   final TextEditingController _emailTextController = TextEditingController();
//   final TextEditingController _userNameTextController = TextEditingController();
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: AppBar(
//         backgroundColor: Colors.transparent,
//         elevation: 0,
//         title: const Text(
//           "Sign Up",
//           style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
//         ),
//       ),
//       body: Container(
//           width: MediaQuery.of(context).size.width,
//           height: MediaQuery.of(context).size.height,
//           decoration: BoxDecoration(
//               gradient: LinearGradient(colors: [
//             hexStringToColor("CB2B93"),
//             hexStringToColor("9546C4"),
//             hexStringToColor("5E61F4")
//           ], begin: Alignment.topCenter, end: Alignment.bottomCenter)),
//           child: SingleChildScrollView(
//               child: Padding(
//             padding: EdgeInsets.fromLTRB(20, 120, 20, 0),
//             child: Column(
//               children: <Widget>[
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 reusableTextField("Enter UserName", Icons.person_outline, false,
//                     _userNameTextController),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 reusableTextField("Enter Email Id", Icons.person_outline, false,
//                     _emailTextController),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 reusableTextField("Enter Password", Icons.lock_outlined, true,
//                     _passwordTextController),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 firebaseUIButton(context, "Sign Up", () {
//                   FirebaseAuth.instance
//                       .createUserWithEmailAndPassword(
//                           email: _emailTextController.text,
//                           password: _passwordTextController.text)
//                       .then((value) {
//                     print("Created New Account");
//                     Navigator.push(context,
//                         MaterialPageRoute(builder: (context) => HomeScreen()));
//                   }).onError((error, stackTrace) {
//                     print("Error ${error.toString()}");
//                   });
//                 })
//               ],
//             ),
//           ))),
//     );
//   }
// }


// import 'package:firebase_auth/firebase_auth.dart';
// import 'home_screen.dart';
// import 'package:flutter_application_1/utils/color_utils.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';


// class SignUpScreen extends StatefulWidget {
//   const SignUpScreen({super.key});

//   @override
//   _SignUpScreenState createState() => _SignUpScreenState();
// }

// class _SignUpScreenState extends State<SignUpScreen> {
//   final TextEditingController _passwordTextController = TextEditingController();
//   final TextEditingController _emailTextController = TextEditingController();
//   final TextEditingController _userNameTextController = TextEditingController();

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: AppBar(
//         backgroundColor: Colors.transparent,
//         elevation: 0,
//         title: const Text(
//           "Sign Up",
//           style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
//         ),
//       ),
//       body: Container(
//         width: MediaQuery.of(context).size.width,
//         height: MediaQuery.of(context).size.height,
//         decoration: BoxDecoration(
//             gradient: LinearGradient(colors: [
//           hexStringToColor("CB2B93"),
//           hexStringToColor("9546C4"),
//           hexStringToColor("5E61F4")
//         ], begin: Alignment.topCenter, end: Alignment.bottomCenter)),
//         child: SingleChildScrollView(
//           child: Padding(
//             padding: EdgeInsets.fromLTRB(20, 120, 20, 0),
//             child: Column(
//               children: <Widget>[
//                 const SizedBox(height: 20),
//                 reusableTextField("Enter UserName", Icons.person_outline, false, _userNameTextController),
//                 const SizedBox(height: 20),
//                 reusableTextField("Enter Email Id", Icons.person_outline, false, _emailTextController),
//                 const SizedBox(height: 20),
//                 reusableTextField("Enter Password", Icons.lock_outlined, true, _passwordTextController),
//                 const SizedBox(height: 20),
//                 firebaseUIButton(context, "Sign Up", () {
//                   FirebaseAuth.instance.createUserWithEmailAndPassword(
//                       email: _emailTextController.text,
//                       password: _passwordTextController.text)
//                       .then((value) {
//                     print("Created New Account");
//                     Navigator.push(context, MaterialPageRoute(builder: (context) => HomeScreen()));
//                   }).onError((error, stackTrace) {
//                     print("Error ${error.toString()}");
//                   });
//                 })
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }


import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'package:flutter_application_1/utils/color_utils.dart';
import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  _SignUpScreenState createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _passwordTextController = TextEditingController();
  final TextEditingController _emailTextController = TextEditingController();
  final TextEditingController _userNameTextController = TextEditingController();

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
                    "Create Account",
                    style: TextStyle(
                      color: Color.fromARGB(255, 51, 85, 255),
                      fontWeight: FontWeight.w900,
                      fontFamily: "Roboto",
                      fontSize: 30,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    "Sign up to get started",
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
                const SizedBox(height: 20),

                // Logo Widget
                logoWidget("assets/images/logo1.png"),
                const SizedBox(height: 30),

                // Sign Up Title
                const Text(
                  "Create Your Account",
                  style: TextStyle(
                    color: Colors.black, // Black text
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  "Join us to get started",
                  style: TextStyle(
                    color: Colors.black54, // Dark gray text
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 30),

                // Username Field
                reusableTextField(
                  "Enter Username",
                  Icons.person_outline,
                  false,
                  _userNameTextController,
                ),
                const SizedBox(height: 20),

                // Email Field
                reusableTextField(
                  "Enter Email Id",
                  Icons.email_outlined,
                  false,
                  _emailTextController,
                ),
                const SizedBox(height: 20),

                // Password Field
                reusableTextField(
                  "Enter Password",
                  Icons.lock_outlined,
                  true,
                  _passwordTextController,
                ),
                const SizedBox(height: 20),

                // Sign Up Button
                firebaseUIButton(context, "Sign Up", () {
                  FirebaseAuth.instance
                      .createUserWithEmailAndPassword(
                    email: _emailTextController.text,
                    password: _passwordTextController.text,
                  )
                      .then((value) {
                    print("Created New Account");
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => HomeScreen()),
                    );
                  }).onError((error, stackTrace) {
                    print("Error ${error.toString()}");
                  });
                }),
                const SizedBox(height: 5),

                // Already have an account? Login
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      "Already have an account?",
                      style: TextStyle(color: Colors.black54), // Dark gray text
                    ),
                    GestureDetector(
                      onTap: () {
                        Navigator.pop(context); // Navigate back to the login screen
                      },
                      child: const Text(
                        " Login",
                        style: TextStyle(
                          color: Color.fromARGB(255, 51, 85, 255), // Blue text for the login link
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 30),
              ])
            ],
          ),
          ),
        ),
    );
  }
}